import os
import sys
import glob
import json
import logging
import time
import argparse
from pathlib import Path
from openai import OpenAI
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("html_cleaning.log"),
        logging.StreamHandler(stream=sys.stdout)
    ]
)
logger = logging.getLogger("html_cleaner")

class HTMLCleaner:
    def __init__(self, api_key: str = os.getenv("OPENAI_API_KEY"), model: str = "deepseek-ai/DeepSeek-V3"):
        """初始化HTML清洗器
        Args:
            api_key: OpenAI API密鑰
            model: 使用的模型，默認為deepseek-ai/DeepSeek-V3
        """
        self.client = OpenAI(base_url=os.getenv("BASE_URL"), api_key=api_key)
        self.model = model

    def parse_output_tags(self, content: str) -> Optional[str]:
        """
        解析<output></output>标签中的内容

        Args:
            content: LLM返回的内容

        Returns:
            解析出的内容，如果没有找到标签或标签不完整则返回None
        """
        import re
        # 使用正则表达式查找<output>与</output>之间的内容
        match = re.search(r'<output>(.*?)</output>', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _clean_code_markers(self, text: str) -> str:
        """移除代码块标记"""
        if text.startswith("```html"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        
        if text.endswith("```"):
            text = text[:-3]
            
        return text.strip()

    def clean_html(self, html_content: str, prompt_template: str, 
                                max_tokens: str = os.getenv("MAX_TOKENS"), 
                                max_attempts: int = 15) -> str:
        """
        使用OpenAI API清洗HTML内容，支持继续生成
        
        Args:
            html_content: 原始HTML内容
            prompt_template: 提示模板
            max_tokens: 最大生成token数
            max_attempts: 最大尝试次数
            
        Returns:
            清洗后的HTML
        """
        try:
            # 构建完整提示
            full_prompt = prompt_template.format(html_content=html_content)
            
            # 初始化消息列表
            messages = [{"role": "user", "content": full_prompt}]
            complete_response = ""
            
            for attempt in range(max_attempts):
                # 调用OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=int(max_tokens)
                )
                
                # 获取本次回应
                current_response = response.choices[0].message.content.strip()
                
                # 将当前回应添加到完整回应中
                complete_response += current_response
                
                # 尝试解析输出标签
                parsed_output = self.parse_output_tags(complete_response)
                
                if parsed_output is not None:
                    # 成功解析，返回结果
                    return parsed_output
                
                # 解析失败，将当前回应作为assistant消息添加到消息列表，继续请求
                messages.append({"role": "assistant", "content": current_response})
                
                logger.info(f"继续生成回应 (第{attempt+1}次尝试)")
            
            # 如果达到最大尝试次数仍未解析成功，返回最佳猜测
            logger.warning(f"无法解析完整的<output>标签，尝试移除代码块标记后返回")
            return self._clean_code_markers(complete_response)
        
        except Exception as e:
            logger.error(f"清洗HTML时出错: {str(e)}")
            raise

    def process_file(self, input_file: str, output_file: str, prompt_template: str, 
                     max_retries: int = 3) -> Dict[str, Any]:
        """處理單個HTML文件，支持重試
        Args:
            input_file: 輸入HTML文件路徑
            output_file: 輸出HTML文件路徑
            prompt_template: 提示模板
            max_retries: 最大重試次數
        Returns:
            處理結果信息字典
        """
        result = {
            "file": input_file,
            "success": False,
            "retries": 0,
            "error": None
        }
        
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # 如果不是第一次嘗試，則記錄重試信息
                if retry_count > 0:
                    logger.info(f"重試處理文件 {input_file} (嘗試 {retry_count}/{max_retries})")
                    
                # 讀取HTML文件
                with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
                    html_content = f.read()
                
                # 清洗HTML
                cleaned_html = self.clean_html(html_content, prompt_template)
                
                # 確保輸出目錄存在
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # 寫入清洗後的HTML
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_html)
                    
                logger.info(f"成功處理: {input_file}" + (f" (重試 {retry_count} 次後)" if retry_count > 0 else ""))
                
                result["success"] = True
                result["retries"] = retry_count
                return result
                
            except Exception as e:
                retry_count += 1
                result["retries"] = retry_count - 1  # 當前已完成的重試次數
                result["error"] = str(e)
                
                if retry_count <= max_retries:
                    logger.warning(f"處理文件失敗 {input_file} (嘗試 {retry_count}/{max_retries}): {str(e)}")
                    # 可以在這裡添加重試前的延遲
                    time.sleep(1)
                else:
                    # 所有重試都失敗了
                    logger.error(f"處理文件最終失敗 {input_file} (已重試 {max_retries} 次): {str(e)}")
                    return result

    def process_directory(self, input_dir: str, output_dir: str, prompt_template: str,
                         file_pattern: str = "*.html", max_files: Optional[int] = None,
                         parallel: int = 1, max_retries: int = 3) -> Dict[str, Any]:
        """批量處理目錄中的HTML文件
        Args:
            input_dir: 輸入目錄
            output_dir: 輸出目錄
            prompt_template: 提示模板
            file_pattern: 文件匹配模式
            max_files: 最大處理文件數
            parallel: 並行處理線程數
            max_retries: 最大重試次數
        Returns:
            處理統計信息
        """
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 獲取所有匹配的HTML文件
        input_files = glob.glob(os.path.join(input_dir, "**", file_pattern), recursive=True)
        if max_files:
            input_files = input_files[:max_files]
        
        total_files = len(input_files)
        logger.info(f"找到 {total_files} 個HTML文件待處理")
        
        # 處理統計
        stats = {
            "total": total_files,
            "success": 0,
            "failed": 0,
            "retried": 0,  # 新增: 統計重試後成功的文件數
            "failed_files": [],  # 新增: 記錄失敗的文件
            "start_time": time.time()
        }
        
        # 並行處理
        if parallel > 1:
            logger.info(f"使用 {parallel} 個線程並行處理")
            def process_file_wrapper(file_info):
                input_file, output_file = file_info
                return self.process_file(input_file, output_file, prompt_template, max_retries)
            
            file_pairs = []
            for input_file in input_files:
                rel_path = os.path.relpath(input_file, input_dir)
                output_file = os.path.join(output_dir, rel_path)
                file_pairs.append((input_file, output_file))
            
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                results = list(executor.map(process_file_wrapper, file_pairs))
            
            # 聚合處理結果
            for result in results:
                if result["success"]:
                    stats["success"] += 1
                    if result["retries"] > 0:
                        stats["retried"] += 1
                else:
                    stats["failed"] += 1
                    stats["failed_files"].append({
                        "file": result["file"],
                        "error": result["error"]
                    })
        else:
            # 串行處理
            for i, input_file in enumerate(input_files):
                # 計算相對路徑以保持目錄結構
                rel_path = os.path.relpath(input_file, input_dir)
                output_file = os.path.join(output_dir, rel_path)
                
                # 處理單個文件
                result = self.process_file(input_file, output_file, prompt_template, max_retries)
                
                if result["success"]:
                    stats["success"] += 1
                    if result["retries"] > 0:
                        stats["retried"] += 1
                else:
                    stats["failed"] += 1
                    stats["failed_files"].append({
                        "file": result["file"],
                        "error": result["error"]
                    })
                
                # 顯示進度
                if (i + 1) % 10 == 0 or (i + 1) == total_files:
                    elapsed = time.time() - stats["start_time"]
                    logger.info(f"進度: {i+1}/{total_files} ({(i+1)/total_files*100:.1f}%) - "
                               f"成功: {stats['success']} (重試成功: {stats['retried']}) - "
                               f"失敗: {stats['failed']} - 用時: {elapsed:.1f}秒")
        
        # 完成統計
        stats["end_time"] = time.time()
        stats["elapsed_seconds"] = stats["end_time"] - stats["start_time"]
        stats["files_per_second"] = stats["total"] / stats["elapsed_seconds"] if stats["elapsed_seconds"] > 0 else 0
        
        logger.info(f"處理完成: 總共 {stats['total']} 個文件, 成功 {stats['success']} (重試成功: {stats['retried']}), 失敗 {stats['failed']}")
        logger.info(f"總用時: {stats['elapsed_seconds']:.1f}秒, 平均速度: {stats['files_per_second']:.2f}文件/秒")
        
        # 如果有失敗的文件，記錄它們
        if stats["failed"] > 0:
            logger.info(f"失敗的文件列表 ({len(stats['failed_files'])}個):")
            for fail_info in stats["failed_files"]:
                logger.info(f"  - {fail_info['file']}: {fail_info['error']}")
        
        return stats

def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='HTML清洗工具 - 使用OpenAI LLM')
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--api-key', type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument('--model', type=str, default=os.getenv("MODEL_ID"))
    parser.add_argument('--file-pattern', type=str, default="*.html")
    parser.add_argument('--max-files', type=int)
    parser.add_argument('--parallel', type=int, default=10)
    parser.add_argument('--prompt', type=str, default="$(cat custom_prompt.txt)")
    parser.add_argument('--max-retries', type=int, default=3)

    args = parser.parse_args()
    
    # 默認提示模板
    default_prompt = """
    """
    
    # 使用自定義提示或默認提示
    prompt_template = args.prompt if args.prompt else default_prompt
    
    # 初始化清洗器
    cleaner = HTMLCleaner(api_key=args.api_key, model=args.model)
    
    # 執行批量處理
    stats = cleaner.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt_template=prompt_template,
        file_pattern=args.file_pattern,
        max_files=args.max_files,
        parallel=args.parallel,
        max_retries=args.max_retries
    )
    
    # 將統計寫入文件
    with open("cleaning_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()
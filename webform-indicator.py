import os
import re
import json
from bs4 import BeautifulSoup
import pandas as pd

# WebForm 特徵標記
webform_indicators = {
    'hidden_fields': [
        '__VIEWSTATE', '__EVENTVALIDATION', '__EVENTTARGET', 
        '__EVENTARGUMENT', '__VIEWSTATEGENERATOR', '__CMSCsrfToken'
    ],
    'js_functions': [
        '__doPostBack', 'WebForm_DoCallback', 'WebForm_InitCallback',
        'WebForm_SaveScrollPosition'
    ],
    'resource_urls': [
        'WebResource.axd', 'ScriptResource.axd'
    ],
    'cms_elements': [
        'CMSPages', 'cmsapi', 'CMS.Application', 'CMSBreadCrumbs'
    ]
}

results = []

# 掃描目錄及所有子目錄中的 HTML 文件
html_directory = "www.towngas.com"

# 先計算要處理的文件總數，用於顯示進度
total_files = 0
for root, dirs, files in os.walk(html_directory):
    for filename in files:
        if filename.endswith('.html'):
            total_files += 1

print(f"找到 {total_files} 個 HTML 文件需處理。")

# 處理計數器
processed_files = 0

# 使用 os.walk 遍歷所有子目錄
for root, dirs, files in os.walk(html_directory):
    for filename in files:
        if filename.endswith('.html'):
            filepath = os.path.join(root, filename)
            processed_files += 1
            
            # 每處理 20 個文件顯示一次進度
            if processed_files % 20 == 0 or processed_files == total_files:
                print(f"正在處理文件 {processed_files}/{total_files}: {filepath}")
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                
                # 初始化統計
                indicators_found = {category: [] for category in webform_indicators}
                
                # 檢查隱藏字段
                for field in webform_indicators['hidden_fields']:
                    if soup.find('input', {'name': field}):
                        indicators_found['hidden_fields'].append(field)
                
                # 檢查 JavaScript 函數
                for func in webform_indicators['js_functions']:
                    if re.search(func, content):
                        indicators_found['js_functions'].append(func)
                
                # 檢查資源 URL
                for resource in webform_indicators['resource_urls']:
                    if resource in content:
                        indicators_found['resource_urls'].append(resource)
                
                # 檢查 CMS 元素
                for element in webform_indicators['cms_elements']:
                    if element in content:
                        indicators_found['cms_elements'].append(element)
                
                # 計算總依賴性分數
                total_indicators = sum(len(v) for v in indicators_found.values())
                coupling_level = "high" if total_indicators > 5 else "middle" if total_indicators > 2 else "low"
                
                # 存儲相對路徑而非僅文件名，更有助於查找文件
                relative_path = os.path.relpath(filepath, html_directory)
                
                results.append({
                    'filepath': relative_path,  # 使用相對路徑而非僅文件名
                    'indicators_count': total_indicators,
                    'coupling_level': coupling_level,
                    'hidden_fields': ', '.join(indicators_found['hidden_fields']),
                    'js_functions': ', '.join(indicators_found['js_functions']),
                    'resource_urls': ', '.join(indicators_found['resource_urls']),
                    'cms_elements': ', '.join(indicators_found['cms_elements'])
                })
            except Exception as e:
                print(f"處理文件 {filepath} 時出錯: {e}")

# 創建分析報告
df = pd.DataFrame(results)
df.sort_values('indicators_count', ascending=False, inplace=True)
df.to_excel('webform_coupling_analysis.xlsx', index=False)
print(f"\n分析完成。處理了 {processed_files} 個 HTML 文件。結果已保存至 'webform_coupling_analysis.xlsx'。")
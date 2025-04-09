# test_reward_server.py
# 测试评分服务器的脚本

import requests
import json
import argparse
import time

# 示例SVG数据
SAMPLE_SVG_PAIRS = [
    {
        "gt_svg_code": """<svg width="100" height="100" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="40" fill="yellow"/>
            <circle cx="35" cy="40" r="5" fill="black"/>
            <circle cx="65" cy="40" r="5" fill="black"/>
            <path d="M30 60 Q50 75 70 60" stroke="black" stroke-width="3" fill="none"/>
        </svg>""",
        "gen_svg_code": """<svg width="100" height="100" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="40" fill="yellow"/>
            <circle cx="35" cy="40" r="5" fill="black"/>
            <circle cx="65" cy="40" r="5" fill="black"/>
            <path d="M30 60 Q50 75 70 60" stroke="black" stroke-width="3" fill="none"/>
        </svg>"""
    },
    {
        "gt_svg_code": """<svg width="100" height="100" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="40" fill="yellow"/>
            <circle cx="35" cy="40" r="5" fill="black"/>
            <circle cx="65" cy="40" r="5" fill="black"/>
            <path d="M30 60 Q50 75 70 60" stroke="black" stroke-width="3" fill="none"/>
        </svg>""",
        "gen_svg_code": """<svg width="100" height="100" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="40" fill="red"/>
            <circle cx="35" cy="40" r="5" fill="black"/>
            <circle cx="65" cy="40" r="5" fill="black"/>
            <path d="M30 70 Q50 55 70 70" stroke="black" stroke-width="3" fill="none"/>
        </svg>"""
    },
    {
        "gt_svg_code": """<svg width="100" height="100" viewBox="0 0 100 100">
            <rect x="20" y="20" width="60" height="60" fill="blue"/>
            <circle cx="50" cy="50" r="20" fill="white"/>
        </svg>""",
        "gen_svg_code": """<svg width="100" height="100" viewBox="0 0 100 100">
            <rect x="20" y="20" width="60" height="60" fill="blue"/>
            <circle cx="50" cy="50" r="15" fill="white"/>
        </svg>"""
    }
]

def test_health(url):
    """测试健康检查端点"""
    try:
        response = requests.get(f"{url}/health")
        print(f"健康检查状态码: {response.status_code}")
        if response.status_code == 200:
            print(f"健康检查响应: {response.json()}")
            return True
        return False
    except Exception as e:
        print(f"健康检查出错: {e}")
        return False

def test_compute_score(url, sample_pairs):
    """测试评分计算端点"""
    results = []
    
    for i, pair in enumerate(sample_pairs):
        try:
            print(f"\n测试示例 {i+1}:")
            start_time = time.time()
            response = requests.post(
                f"{url}/compute_score",
                json=pair,
                headers={"Content-Type": "application/json"}
            )
            request_time = time.time() - start_time
            
            print(f"请求耗时: {request_time:.4f}秒")
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"评分结果: {json.dumps(result, indent=2)}")
                results.append(result)
            else:
                print(f"错误: {response.text}")
                results.append({"error": response.text})
        except Exception as e:
            print(f"请求出错: {e}")
            results.append({"error": str(e)})
    
    return results

def main():
    parser = argparse.ArgumentParser(description='测试评分服务器')
    parser.add_argument('--url', type=str, default='http://127.0.0.1:5000', help='评分服务器URL')
    args = parser.parse_args()
    
    print(f"测试评分服务器: {args.url}")
    
    # 测试健康检查
    if not test_health(args.url):
        print("健康检查失败，退出测试")
        return
    
    # 测试评分计算
    print("\n测试评分计算...")
    results = test_compute_score(args.url, SAMPLE_SVG_PAIRS)
    
    # 显示摘要
    print("\n测试摘要:")
    for i, result in enumerate(results):
        if "error" in result:
            print(f"示例 {i+1}: 失败 - {result['error']}")
        else:
            print(f"示例 {i+1}: 成功 - 总分: {result.get('total_score', 'N/A')}")

if __name__ == "__main__":
    main()
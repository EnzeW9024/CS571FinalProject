"""
自动化测试脚本
运行一系列测试验证代码功能
"""

import subprocess
import sys
import time

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"测试: {description}")
    print(f"命令: {cmd}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✓ 成功 (耗时: {elapsed:.2f}秒)")
        if result.stdout:
            print(result.stdout)
        return True
    else:
        print(f"✗ 失败 (耗时: {elapsed:.2f}秒)")
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
        return False

def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("开始自动化测试")
    print("="*60)
    
    tests = [
        ("python test_basic.py", "基本功能测试"),
        ("python main.py --matches 5 --rollouts 10", "快速功能验证（5场比赛）"),
        ("python main.py --matches 10 --rollouts 20 --opponent tight", "测试Tight对手"),
        ("python main.py --matches 10 --rollouts 20 --opponent mixed", "测试Mixed对手"),
    ]
    
    results = []
    for cmd, desc in tests:
        success = run_command(cmd, desc)
        results.append((desc, success))
        time.sleep(1)  # 短暂暂停
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for desc, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status}: {desc}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""
五子棋AI训练质量测试脚本
测试AI是否真的学到了东西，还是只是在拟合噪声
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple

from ai.network import PolicyValueNetwork, PolicyValueNetworkSmall
from ai.mcts import MCTS, MCTSPlayer, RandomPlayer
from game.board import Board

class AIQualityTester:
    """AI质量测试器"""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: 模型文件路径
        """
        print(f"加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 判断网络类型
        num_res_blocks = checkpoint.get('num_res_blocks', 10)
        if num_res_blocks <= 5:
            self.network = PolicyValueNetworkSmall()
        else:
            self.network = PolicyValueNetwork()
        
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.eval()
        print(f"✅ 模型加载成功 (参数量: {self.network.count_parameters():,})")
        print()
    
    def test_all(self):
        """运行所有测试"""
        print("=" * 70)
        print(" 五子棋AI训练质量测试")
        print("=" * 70)
        print()
        
        results = {}
        
        # 测试1：梯度健康检查
        print("🔍 测试1: 梯度健康检查")
        print("-" * 70)
        results['gradient'] = self.test_gradient_health()
        print()
        
        # 测试2：策略质量检查
        print("🎯 测试2: 策略质量检查")
        print("-" * 70)
        results['policy'] = self.test_policy_quality()
        print()
        
        # 测试3：价值准确性检查
        print("💎 测试3: 价值评估准确性")
        print("-" * 70)
        results['value'] = self.test_value_accuracy()
        print()
        
        # 测试4：基本策略检查
        print("🧠 测试4: 基本五子棋策略")
        print("-" * 70)
        results['strategy'] = self.test_basic_strategy()
        print()
        
        # 测试5：实战能力检查
        print("⚔️ 测试5: 实战能力")
        print("-" * 70)
        results['combat'] = self.test_combat_ability()
        print()
        
        # 总体评分
        self.print_summary(results)
    
    def test_gradient_health(self) -> dict:
        """测试梯度是否健康"""
        self.network.train()
        
        # 创建随机测试数据
        batch_size = 32
        states = torch.randn(batch_size, 3, 15, 15)
        target_probs = torch.rand(batch_size, 225)
        target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)
        target_values = torch.rand(batch_size, 1) * 2 - 1  # [-1, 1]
        
        # 前向传播
        pred_logits, pred_values = self.network(states)
        
        # 计算损失
        policy_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(pred_logits, dim=1),
            target_probs,
            reduction='batchmean'
        )
        value_loss = torch.nn.functional.mse_loss(pred_values, target_values)
        loss = 3.0 * value_loss + policy_loss
        
        # 反向传播
        self.network.zero_grad()
        loss.backward()
        
        # 收集梯度统计
        grad_norms = []
        grad_means = []
        for name, param in self.network.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
                grad_means.append(param.grad.abs().mean().item())
        
        mean_norm = np.mean(grad_norms)
        max_norm = np.max(grad_norms)
        min_norm = np.min(grad_norms)
        mean_abs = np.mean(grad_means)
        
        print(f"  梯度范数统计:")
        print(f"    平均: {mean_norm:.6f}")
        print(f"    最大: {max_norm:.6f}")
        print(f"    最小: {min_norm:.6f}")
        print(f"  梯度绝对值平均: {mean_abs:.6f}")
        
        # 判断
        issues = []
        if max_norm > 100:
            print(f"  ❌ 警告: 梯度爆炸 (最大={max_norm:.2f} > 100)")
            issues.append('gradient_explosion')
        elif max_norm > 10:
            print(f"  ⚠️  注意: 梯度偏大 (最大={max_norm:.2f} > 10)")
            issues.append('gradient_large')
        else:
            print(f"  ✅ 梯度范围正常 (最大={max_norm:.2f} < 10)")
        
        if mean_abs < 1e-6:
            print(f"  ❌ 警告: 梯度消失 (平均={mean_abs:.2e} < 1e-6)")
            issues.append('gradient_vanishing')
        elif mean_abs < 1e-4:
            print(f"  ⚠️  注意: 梯度偏小 (平均={mean_abs:.2e})")
            issues.append('gradient_small')
        else:
            print(f"  ✅ 梯度大小正常 (平均={mean_abs:.2e})")
        
        self.network.eval()
        
        return {
            'mean_norm': mean_norm,
            'max_norm': max_norm,
            'mean_abs': mean_abs,
            'issues': issues,
            'pass': len(issues) == 0
        }
    
    def test_policy_quality(self) -> dict:
        """测试策略质量"""
        board = Board()
        state = board.encode_state()
        
        # 测试1：初始局面偏好中心
        probs, value = self.network.predict(state)
        probs_2d = probs.reshape(15, 15)
        
        # 中心区域 (5x5) - 用Python原生计算避免numpy问题
        center_probs = float(sum(probs_2d[i, j] for i in range(5, 10) for j in range(5, 10)))
        # 边缘区域
        edge_probs = float(
            sum(probs_2d[0, j] for j in range(15)) +
            sum(probs_2d[14, j] for j in range(15)) +
            sum(probs_2d[i, 0] for i in range(1, 14)) +  # 避免重复角落
            sum(probs_2d[i, 14] for i in range(1, 14))
        )
        
        print(f"  初始局面概率分布:")
        print(f"    中心5x5区域: {center_probs:.4f}")
        print(f"    边缘区域: {edge_probs:.4f}")
        # 转换为Python原生类型避免numpy问题
        probs_list = probs.tolist() if hasattr(probs, 'tolist') else list(probs)
        max_prob = max(probs_list)
        
        # 计算熵
        entropy = -sum(p * np.log(p + 1e-10) for p in probs_list if p > 0)
        
        print(f"    最大概率: {max_prob:.4f}")
        print(f"    熵: {entropy:.4f} (均匀=5.42)")
        
        issues = []
        if center_probs < 0.3:
            print(f"  ⚠️  AI不够偏好中心 (中心概率<30%)")
            issues.append('not_center_biased')
        else:
            print(f"  ✅ AI偏好中心落子")
        
        if max_prob < 0.01:
            print(f"  ❌ 策略太平均，没有明显偏好")
            issues.append('too_uniform')
        else:
            print(f"  ✅ 有明显的落子偏好")
        
        # 测试2：简单局面的策略
        board.move(7, 7)  # 中心
        board.move(7, 8)  # 右边
        state = board.encode_state()
        probs, value = self.network.predict(state)
        probs_2d = probs.reshape(15, 15)
        
        # 检查是否偏好关键位置 (7,6) 或 (7,9)
        key_positions = [(7, 6), (7, 9)]
        key_probs = float(sum(probs_2d[x, y] for x, y in key_positions))
        
        print(f"\n  简单对局测试 (黑7,7 白7,8):")
        print(f"    关键位置概率: {key_probs:.4f}")
        
        if key_probs < 0.1:
            print(f"  ⚠️  没识别出关键位置")
            issues.append('no_key_position')
        else:
            print(f"  ✅ 能识别关键位置")
        
        return {
            'center_bias': center_probs,
            'edge_prob': edge_probs,
            'max_prob': max_prob,
            'key_prob': key_probs,
            'issues': issues,
            'pass': len(issues) <= 1
        }
    
    def test_value_accuracy(self) -> dict:
        """测试价值评估准确性"""
        test_cases = []
        
        # 测试1：空棋盘 (应该接近0)
        board = Board()
        state = board.encode_state()
        _, value = self.network.predict(state)
        test_cases.append(('空棋盘', value, 0.0, 0.3))
        
        # 测试2：黑方优势局面
        board = Board()
        board.move(7, 7)
        board.move(6, 7)
        board.move(7, 8)
        board.move(6, 8)
        board.move(7, 9)  # 黑方三连
        state = board.encode_state()
        _, value = self.network.predict(state)
        test_cases.append(('黑三连', value, 0.3, None))  # 应该>0.3
        
        # 测试3：白方优势局面
        board = Board()
        board.move(5, 5)
        board.move(7, 7)
        board.move(5, 6)
        board.move(7, 8)
        board.move(6, 5)
        board.move(7, 9)  # 白方三连
        state = board.encode_state()
        _, value = self.network.predict(state)
        test_cases.append(('白三连', value, None, -0.3))  # 应该<-0.3
        
        print(f"  价值评估测试:")
        passed = 0
        for name, actual, target_min, target_max in test_cases:
            if target_min is not None and target_max is not None:
                # 应该在范围内
                if target_min <= actual <= target_max:
                    status = "✅"
                    passed += 1
                else:
                    status = "❌"
                print(f"    {status} {name}: {actual:.4f} (期望: {target_min:.2f}~{target_max:.2f})")
            elif target_min is not None:
                # 应该大于阈值
                if actual > target_min:
                    status = "✅"
                    passed += 1
                else:
                    status = "❌"
                print(f"    {status} {name}: {actual:.4f} (期望: >{target_min:.2f})")
            else:
                # 应该小于阈值
                if actual < target_max:
                    status = "✅"
                    passed += 1
                else:
                    status = "❌"
                print(f"    {status} {name}: {actual:.4f} (期望: <{target_max:.2f})")
        
        print(f"\n  通过: {passed}/{len(test_cases)}")
        
        return {
            'passed': passed,
            'total': len(test_cases),
            'pass': passed >= len(test_cases) - 1  # 允许1个失败
        }
    
    def test_basic_strategy(self) -> dict:
        """测试基本五子棋策略"""
        tests = []
        
        # 测试1：能否识别需要防守
        print("  测试1: 识别防守")
        board = Board()
        board.move(7, 7)
        board.move(5, 5)
        board.move(7, 8)
        board.move(5, 6)
        board.move(7, 9)  # 黑方三连
        # 现在轮到白方，应该下在(7,10)或(7,6)防守
        
        mcts = MCTS(self.network, simulations=100, c_puct=1.0)
        probs = mcts.get_action_probs(board, temperature=0, add_noise=False)
        best_action = np.argmax(probs)
        best_x, best_y = best_action // 15, best_action % 15
        
        defense_positions = [(7, 10), (7, 6)]
        is_defense = (best_x, best_y) in defense_positions
        
        print(f"    AI选择: ({best_x}, {best_y})")
        print(f"    防守位置: {defense_positions}")
        if is_defense:
            print(f"    ✅ 选择了防守位置")
            tests.append(True)
        else:
            print(f"    ❌ 没有选择防守")
            tests.append(False)
        
        # 测试2：能否识别进攻机会
        print("\n  测试2: 识别进攻")
        board = Board()
        board.move(7, 7)
        board.move(6, 7)
        board.move(7, 8)
        board.move(6, 8)
        # 现在轮到黑方，(7,9)可以形成三连
        
        mcts = MCTS(self.network, simulations=100, c_puct=1.0)
        probs = mcts.get_action_probs(board, temperature=0, add_noise=False)
        probs_2d = probs.reshape(15, 15)
        
        attack_prob = probs_2d[7, 9]
        print(f"    进攻位置(7,9)概率: {attack_prob:.4f}")
        
        # 获取top3位置
        top3_indices = np.argsort(probs)[-3:]
        top3_positions = [(idx // 15, idx % 15) for idx in top3_indices]
        print(f"    AI偏好的top3: {top3_positions}")
        
        if (7, 9) in top3_positions:
            print(f"    ✅ 进攻位置在top3中")
            tests.append(True)
        else:
            print(f"    ⚠️  进攻位置不在top3")
            tests.append(False)
        
        # 测试3：避免明显的坏棋
        print("\n  测试3: 避免坏棋")
        board = Board()
        board.move(7, 7)
        
        mcts = MCTS(self.network, simulations=100, c_puct=1.0)
        probs = mcts.get_action_probs(board, temperature=0, add_noise=False)
        probs_2d = probs.reshape(15, 15)
        
        # 角落和边缘不应该有高概率
        corners = [(0, 0), (0, 14), (14, 0), (14, 14)]
        corner_probs = float(sum(probs_2d[x, y] for x, y in corners))
        
        print(f"    四个角落概率之和: {corner_probs:.4f}")
        if corner_probs < 0.05:
            print(f"    ✅ 避开了角落")
            tests.append(True)
        else:
            print(f"    ⚠️  还会考虑角落")
            tests.append(False)
        
        passed = sum(tests)
        print(f"\n  策略测试通过: {passed}/{len(tests)}")
        
        return {
            'passed': passed,
            'total': len(tests),
            'tests': tests,
            'pass': passed >= 2  # 至少通过2个
        }
    
    def test_combat_ability(self) -> dict:
        """测试实战能力"""
        print("  vs 随机玩家 (10局)")
        
        player = MCTSPlayer(self.network, simulations=100, c_puct=1.0, temperature=0)
        random_player = RandomPlayer()
        
        wins = 0
        losses = 0
        draws = 0
        
        for game_idx in range(10):
            board = Board()
            
            # 交替先后手
            if game_idx % 2 == 0:
                players = [player, random_player]
                ai_color = 1
            else:
                players = [random_player, player]
                ai_color = 2
            
            current = 0
            moves = 0
            while not board.is_game_over() and moves < 225:
                action = players[current].get_action(board)
                x, y = action // 15, action % 15
                board.move(x, y)
                current = 1 - current
                moves += 1
            
            winner = board.get_winner()
            if winner == ai_color:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1
        
        win_rate = wins / 10 * 100
        print(f"    胜: {wins}, 负: {losses}, 平: {draws}")
        print(f"    胜率: {win_rate:.1f}%")
        
        if win_rate >= 90:
            print(f"    ✅ 优秀！完全碾压随机")
            grade = 'excellent'
        elif win_rate >= 70:
            print(f"    ✅ 良好！明显强于随机")
            grade = 'good'
        elif win_rate >= 50:
            print(f"    ⚠️  一般，略强于随机")
            grade = 'fair'
        else:
            print(f"    ❌ 较弱，接近随机水平")
            grade = 'poor'
        
        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': win_rate,
            'grade': grade,
            'pass': win_rate >= 70
        }
    
    def print_summary(self, results: dict):
        """打印总体评分"""
        print("=" * 70)
        print(" 综合评估")
        print("=" * 70)
        print()
        
        # 计算各项得分
        scores = {
            '梯度健康': 100 if results['gradient']['pass'] else 
                       (50 if 'gradient_large' in results['gradient']['issues'] else 0),
            '策略质量': 100 if results['policy']['pass'] else 60,
            '价值准确性': results['value']['passed'] / results['value']['total'] * 100,
            '基本策略': results['strategy']['passed'] / results['strategy']['total'] * 100,
            '实战能力': results['combat']['win_rate']
        }
        
        total_score = sum(scores.values()) / len(scores)
        
        print(f"各项得分:")
        for name, score in scores.items():
            bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
            print(f"  {name:<12} [{bar}] {score:5.1f}%")
        
        print(f"\n总体得分: {total_score:.1f}/100")
        print()
        
        # 等级评定
        if total_score >= 90:
            grade = "S"
            comment = "🌟 优秀！AI已经学会了五子棋策略，可以部署使用！"
        elif total_score >= 80:
            grade = "A"
            comment = "✅ 良好！AI有基本策略，可以继续训练提升。"
        elif total_score >= 70:
            grade = "B"
            comment = "🟡 及格！AI开始学习，建议继续训练。"
        elif total_score >= 60:
            grade = "C"
            comment = "⚠️  较弱！AI学习不足，需要继续训练。"
        else:
            grade = "D"
            comment = "❌ 较差！AI可能没有学到东西，需要检查训练配置。"
        
        print(f"等级评定: {grade}")
        print(f"{comment}")
        print()
        
        # 具体建议
        print("改进建议:")
        suggestions = []
        
        if not results['gradient']['pass']:
            if 'gradient_explosion' in results['gradient']['issues']:
                suggestions.append("- ⚠️  梯度爆炸！立即降低学习率到0.00001")
            elif 'gradient_large' in results['gradient']['issues']:
                suggestions.append("- ⚠️  梯度偏大，建议降低学习率")
        
        if not results['policy']['pass']:
            if 'too_uniform' in results['policy']['issues']:
                suggestions.append("- 策略太平均，增加训练轮数或MCTS模拟次数")
            if 'not_center_biased' in results['policy']['issues']:
                suggestions.append("- AI没学会偏好中心，检查训练数据质量")
        
        if not results['value']['pass']:
            suggestions.append("- 价值评估不准，可能需要更多训练数据")
        
        if not results['strategy']['pass']:
            suggestions.append("- 基本策略缺失，继续训练或增加MCTS模拟次数")
        
        if results['combat']['win_rate'] < 70:
            suggestions.append("- 实战能力弱，建议续训50-100轮")
        
        if not suggestions:
            suggestions.append("- ✅ 当前状态良好，可以继续训练或部署使用！")
        
        for suggestion in suggestions:
            print(suggestion)
        
        print()
        print("=" * 70)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='五子棋AI训练质量测试')
    parser.add_argument('--model', type=str, default='models/checkpoint_65.pth',
                      help='模型文件路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ 错误：找不到模型文件 {args.model}")
        print("\n可用的模型文件：")
        model_dir = Path('models')
        if model_dir.exists():
            for f in sorted(model_dir.glob('*.pth')):
                print(f"  - {f}")
        return
    
    tester = AIQualityTester(args.model)
    tester.test_all()


if __name__ == '__main__':
    main()
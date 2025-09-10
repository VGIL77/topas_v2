#!/bin/bash
echo "🚀 Enhanced TOPAS Training Monitor - Watching for breakthrough features..."
echo "======================================================================"

echo "📊 Training Progress:"
tail -f training_enhanced_phased.log 2>/dev/null | strings | grep -E "Epoch.*complete.*EM=|Dream.*cycle|self-play.*Generated|best_em_|best_acc_|📊.*checkpoint" --line-buffered &

echo "🧠 Dream System Status:"
tail -f training_enhanced_phased.log 2>/dev/null | strings | grep -E "Dream-Trainer.*Triggering|Dream.*Full cycle|buffer_len|num_themes|nmda_loss" --line-buffered &

echo "🎮 Self-Play Activity:"
tail -f training_enhanced_phased.log 2>/dev/null | strings | grep -E "🎮.*Generated|SelfPlayBuffer|generate_batch" --line-buffered &

echo "💾 Checkpoints & Metrics:"
watch -n 10 "ls -la *.pt | grep -E 'best_em_|best_acc_|eval_epoch' | tail -5" &

echo "🎯 Monitoring active. Press Ctrl+C to stop all monitors."
wait
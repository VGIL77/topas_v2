#!/bin/bash
# Monitor training progress with RelMem

echo "=== Training Monitor with RelMem ==="
echo ""

# Check if training is running
PID=$(ps aux | grep train_simple_hrm_topas | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$PID" ]; then
    echo "✅ Training is running (PID: $PID)"
else
    echo "❌ Training is not running"
fi
echo ""

# Get latest epoch info
echo "📊 Latest Epoch Status:"
grep "Epoch.*complete" training_relmem_enhanced.log | tail -5
echo ""

# Check current progress
echo "📈 Current Progress:"
tail -1 training_relmem_enhanced.log | grep -E "Epoch [0-9]+"
echo ""

# Check exact match rate
echo "🎯 Exact Match Progress:"
grep "EM=" training_relmem_enhanced.log | grep -v "EM=0.00%" | tail -5
if [ $? -ne 0 ]; then
    echo "No exact matches achieved yet"
fi
echo ""

# Check RelMem metrics
echo "🧠 RelMem Activity:"
echo -n "Inverse loss applications: "
grep "relmem_inverse_loss" training_relmem_enhanced.log | wc -l
echo -n "Op bias merges: "
grep "op_bias merged" training_relmem_enhanced.log | wc -l
echo ""

# Check for any errors
echo "⚠️  Recent Warnings/Errors:"
grep -E "WARNING|ERROR|error|failed" training_relmem_enhanced.log | tail -5
if [ $? -ne 0 ]; then
    echo "No recent errors"
fi
echo ""

# Estimate time remaining
CURRENT_EPOCH=$(tail -100 training_relmem_enhanced.log | grep -oE "Epoch [0-9]+" | tail -1 | awk '{print $2}')
if [ -n "$CURRENT_EPOCH" ]; then
    REMAINING=$((150 - CURRENT_EPOCH))
    echo "⏱️  Estimated Progress: Epoch $CURRENT_EPOCH/150 ($REMAINING epochs remaining)"
fi
#!/bin/bash
echo "Running pre-hooks before committing..."

echo "======FORMAT====="
black . -q

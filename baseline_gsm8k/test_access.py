#!/usr/bin/env python3
from transformers import AutoTokenizer
from huggingface_hub import login, whoami
import sys

print("=" * 70)
print("CHECKING LLAMA ACCESS")
print("=" * 70)

# Login with stored token
try:
    login()
    user = whoami()
    print(f"\n✅ Logged in as: {user['name']}")
except Exception as e:
    print(f"\n❌ Login failed: {e}")
    sys.exit(1)

# Test Meta-Llama-3.1-8B access
print("\n" + "-" * 70)
print("Testing Meta-Llama-3.1-8B...")
print("-" * 70)

try:
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B",
        token=True
    )
    print("✅ SUCCESS! You have access to Meta-Llama-3.1-8B")
except Exception as e:
    print(f"❌ FAILED! Error: {str(e)[:200]}")
    print("\n🔧 TO FIX:")
    print("   Visit: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B")
    print("   Click 'Request Access' and accept the license")
    sys.exit(1)

# Test Instruct model
print("\n" + "-" * 70)
print("Testing Meta-Llama-3.1-8B-Instruct...")
print("-" * 70)

try:
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        token=True
    )
    print("✅ SUCCESS! You have access to Meta-Llama-3.1-8B-Instruct")
except Exception as e:
    print(f"❌ FAILED! Error: {str(e)[:200]}")

print("\n" + "=" * 70)
print("CHECKS COMPLETE")
print("=" * 70)

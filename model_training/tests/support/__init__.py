"""
Expose shared synthetic task builders for lifecycle and contract tests.

Support objects model the public TaskSpec surface while remaining unregistered and
small enough for CPU unit tests. They are intentionally not storage fixtures,
scientific benchmark tasks, or production defaults; individual modules document
additional simplifications.
"""

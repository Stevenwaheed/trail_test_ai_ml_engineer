"""
Task 3 -- A few quick assertions proving permission enforcement works.

This is deliberately light (2-3 checks), not a full test suite -- the
point is the hard requirement they prove, not test-suite completeness.
"""

import asyncio
from app.retrieval import retrieve
from app.users import TEST_USERS


import pytest
import asyncio
from app.retrieval import retrieve
from app.users import TEST_USERS

# Use a single event loop to prevent the global AsyncOpenAI client
# from crashing when switching between tests.
loop = asyncio.new_event_loop()

def test_different_users_get_differently_scoped_results():
    async def run_test():
        query = "Q3 2026 financial summary revenue expenses maintenance log HVAC repair"
        
        user_a_chunks = await retrieve(query, TEST_USERS["user_a"], top_k=5)
        user_b_chunks = await retrieve(query, TEST_USERS["user_b"], top_k=5)
        
        user_a_ids = {c.doc_id for c in user_a_chunks}
        user_b_ids = {c.doc_id for c in user_b_chunks}
        
        assert user_a_ids != user_b_ids
        
        assert 0 in user_a_ids or 3 in user_a_ids
        assert 2 not in user_a_ids
        
        assert 2 in user_b_ids
        assert 0 not in user_b_ids

    loop.run_until_complete(run_test())


def test_all_only_user_never_sees_restricted_chunks():
    async def run_test():
        query = "Q3 2026 financial summary revenue expenses maintenance log HVAC holiday schedule"
        
        user_c_chunks = await retrieve(query, TEST_USERS["user_c"], top_k=5)
        user_c_ids = {c.doc_id for c in user_c_chunks}
        
        assert 0 not in user_c_ids
        assert 2 not in user_c_ids
        assert 1 in user_c_ids

    loop.run_until_complete(run_test())

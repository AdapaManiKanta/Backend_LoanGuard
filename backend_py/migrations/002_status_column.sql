-- ============================================================
-- LoanGuard Migration 002: Application Status Column
-- Run in Supabase SQL Editor
-- ============================================================

ALTER TABLE applications
  ADD COLUMN IF NOT EXISTS status VARCHAR(30) DEFAULT 'Pending'
    CHECK (status IN ('Pending', 'Under Review', 'Approved', 'Rejected'));

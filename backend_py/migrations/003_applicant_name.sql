-- ============================================================
-- LoanGuard Migration 003: Applicant Name Column
-- Run in Supabase SQL Editor
-- ============================================================

ALTER TABLE applications
  ADD COLUMN IF NOT EXISTS applicant_name VARCHAR(150) DEFAULT '';

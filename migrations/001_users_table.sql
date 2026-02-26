-- ============================================================
-- LoanGuard: Users Table Migration
-- Run on your Supabase PostgreSQL instance
-- ============================================================

CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(50)  UNIQUE NOT NULL,
    name          VARCHAR(100) NOT NULL,
    email         VARCHAR(150) UNIQUE,
    password_hash TEXT         NOT NULL,
    role          VARCHAR(20)  NOT NULL DEFAULT 'OFFICER'
                               CHECK (role IN ('OFFICER', 'MANAGER', 'ADMIN')),
    is_active     BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMP    DEFAULT NOW(),
    last_login    TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- ============================================================
-- Audit log table (if not yet created)
-- ============================================================
CREATE TABLE IF NOT EXISTS audit_log (
    id           SERIAL PRIMARY KEY,
    event_type   VARCHAR(50),
    username     VARCHAR(100),
    details      JSONB,
    created_at   TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- NOTE: After running this script, seed the admin user.
-- Run:  python generate_hash.py <your_password>
-- Then INSERT manually, or let the app seed on first boot.
-- ============================================================

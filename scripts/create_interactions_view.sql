-- scripts/create_interactions_view.sql
-- This script creates a DuckDB VIEW to alias book_id as item_id for pipeline compatibility
-- Should be run at the start of orchestrator.py or any session that expects 'item_id' in 'interactions'.

DROP VIEW IF EXISTS interactions;

CREATE VIEW interactions AS
SELECT
    *,
    book_id AS item_id
FROM curated_reviews;

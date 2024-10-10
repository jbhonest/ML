SELECT
    pg_size_pretty(pg_table_size('articles')) AS table_size,
    pg_size_pretty(pg_indexes_size('articles')) AS indexes_size,
    pg_size_pretty(pg_total_relation_size('articles')) AS total_size
;

-- Wheres the Glams
SELECT band_name, YEAR(CURRENT_TIMESTAMP) - formed as lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC;
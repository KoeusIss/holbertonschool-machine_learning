-- Wheres the Glams
SELECT band_name, IFNULL(split, 2021) - formed as lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC;

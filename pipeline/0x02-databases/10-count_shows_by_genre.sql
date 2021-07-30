-- Lists all genres with number of shows
SELECT tv_genres.name as genre, COUNT(*) AS number_of_shows
FROM tv_genres
	LEFT JOIN tv_show_genres
		ON tv_genres.id = tv_show_genres.genre_id
GROUP BY genre
ORDER BY number_of_shows DESC;
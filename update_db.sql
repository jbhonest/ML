update articles
set category = 'economics'
where category = 'v2';

select category, count(*) from articles
group by category;



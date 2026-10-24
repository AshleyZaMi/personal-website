import express from 'express';
import db from './db.js'

const app = express();
const port = 3000;

app.use(express.urlencoded({extended: false}));

app.set('view engine', 'ejs');
app.set('views', 'views');

// did all connections in db.js file
app.get('/', async (req, res) => {

  try {

    const sql = `
      SELECT *
      FROM song_db.genres;
    `;

    const sqlMediaTypes = `
      SELECT *
      FROM song_db.media_types;
    `;

    const results = await db.query( sql );

    const resultsMediaTypes = await db.query(sqlMediaTypes);

    res.render('searchForm', {
      genres: results.rows,
      mediaTypes: resultsMediaTypes.rows
    });

  } catch (err) {
    console.log(err);
    res.send('Server Error');
  }

});

app.get('/searchResults', async (req, res) => {

  try {

    // SQL for search results to be displayed.
    let sql = `
      SELECT tracks.track_id,
        tracks.name AS track,
        genres.name AS genre,
        media_types.name AS media
      FROM song_db.tracks
      LEFT JOIN song_db.genres
        ON genres.genre_id = tracks.genre_id
      LEFT JOIN song_db.media_types
        ON media_types.media_type_id = tracks.media_type_id
      WHERE 1 = 1
    `;

    // SQL to get the total number of results before pagination.
    // here getting the aggregate cause just want one row, to knot use many results
    let sqlCount = `
      SELECT COUNT(*)
      FROM song_db.tracks
      WHERE 1 = 1
    `;

    // This is where we can conditionally add search criteria. (like what user put so using prepared statements to put those in)
    const userInputs = []
    let preparedIndex = 1; 
    const trackInput = req.query.trackName;

    if ( typeof(trackInput) != 'undefined' && trackInput.length > 0 ) {
      // Users typed "Track Name" into search form.
      // Append "track name" into the userInputs prepared statement array.
      sql = sql + ` AND tracks.name ILIKE $${preparedIndex}`;
      sqlCount = sqlCount + ` AND tracks.name ILIKE $${preparedIndex}`;
      userInputs.push(`'%${trackInput}%'`)
      ++preparedIndex
    }

    const genreIdInput = req.query.genreId;

    if (typeof(genreIdInput) != 'undefined' && genreIdInput.length > 0) {
      sql = sql + ` AND tracks.genre_id = $${preparedIndex}`;
      sqlCount = sqlCount + ` AND tracks.genre_id = $${preparedIndex}`;
      userInputs.push(genreIdInput)
      ++preparedIndex
    }

    const mediaIdInput = req.query.mediaTypeId;
    // make sure to have dollarsign up to kdo prepared statement
    if (typeof(mediaIdInput) != 'undefined' && mediaIdInput.length > 0) {
      sql = sql + ` AND tracks.media_type_id = $${preparedIndex}`;
      sqlCount = sqlCount + ` AND tracks.media_type_id = $${preparedIndex}`;
      userInputs.push(mediaIdInput)
      ++preparedIndex
    }

    // pass in userinputs
    const resultsCount = await db.query( sqlCount, userInputs );

    console.log(resultsCount)

    // HERE FOR LAST PAGE: in order to know what the last page is we need to know:
    // 1. How many total result we have(which is resultsCount)
    //2. How many results per page do we display
    const totalResults = resultsCount.rows[0].count //will be count
    // 2.
    // here hardcoind decide what you want
    const limit = 10
    // divide total results by limit count and then ceinling of the leftover
    const lastPage = Math.ceil(totalResults/limit)

    console.log(lastPage)

    // will do simialr pagination to amazon where it has page number, if do &page=1 will see this "{ trackName: '', genreId: '', mediaTypeId: '', page: '1' }"
    // console.log(req.query)

    // want to parse page to make sure its a valid page number. Otherwise will not allow it
    let page = parseInt(req.query.page)

    // if it is not a number will default page to 1
     // here to prevent the negative numbers too
    if(isNaN(page) || page < 1){
      page=1

     
    }else if(page > lastPage){
      // default the page to be the last page so they wouldn't be able to click next and stay on actual last page
      page = lastPage
    }

    //console.log(page)//will go into the if loop and will default to none, isNaN will only avvept if number


    // build firstpage url and return all query parameters
    // ex. from page 100 will set it to 1 when clicked on first
    let firstPageUrl = req.query
    // additional property
    firstPageUrl.page = 1

    console.log(firstPageUrl)

    // turn this javascropt object into the search parameter that will turn to the url
    firstPageUrl = new URLSearchParams(firstPageUrl).toString()

    //console.log(firstPageUrl)//would print out trackName=&genreId=&mediaTypeId=&page=1

    // for previous page
    let prevPageUrl = req.query
    prevPageUrl.page = page -1
    prevPageUrl = new URLSearchParams(prevPageUrl).toString()

    // for next page
    let nextPageUrl = req.query
    nextPageUrl.page = page + 1
    nextPageUrl = new URLSearchParams(nextPageUrl).toString()

    // for last will have to do some math to know how many records we have.
    // for next page
    let lastPageUrl = req.query
    lastPageUrl.page = lastPage
    lastPageUrl = new URLSearchParams(lastPageUrl).toString()

    // only showing the top, so if they want more then can do it with pagination
    // sql = sql + ` LIMIT 10`;
    // will not hardcode
    const offset =(page -1) * limit //limit is results per page we have 
    sql = sql + ` LIMIT ${limit} OFFSET ${offset}`;

    console.log(sql)
    console.log(userInputs)
    // console.log(sql);

    // pass in userINputs
    const results = await db.query( sql, userInputs);

    res.render('searchResults', {
      tracks: results.rows,
      // lets go ahead and show the page it currently is on
      currentPage: page,
      // if key is the same as the object i am passing i will just admit that and will have the same effect
      
      // firstPageUrl: firstPageUrl
      firstPageUrl,
      prevPageUrl,
      // passing to front end
      nextPageUrl,
      lastPageUrl,
      // here passing in totalResults so that it replaces X in the front end
      totalResults

    });


  } catch (err) {
    console.log(err);
    res.send('Server Error');
  }

  
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
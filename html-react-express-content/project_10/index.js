import express from 'express';

// whenever connect it to the program
// Whole project relies on connecting in a database
import pg from 'pg';

// here will make the app run
// starts the server for us with express()
const app = express()
// the port we want it to run
const port = 3000;

// will use the pool -> is another name for connection to the database
const pool = new pg.Pool({
    host: process.env.DB_HOST,
    port: process.env.DP_PORT,
    user: process.env.DB_USER,
    password: process.env.DB_PASS,
    database: process.env.DB_NAME
})

pool.on('connect', async(client) => {
    try{
        await client.query(`SET search_path TO ${process.env.DB_SCHEMA}`)
    }
    catch(err){
        console.log('Error setting search_path:', err)
    }
    
})

// set the view, have to do this to use EJS to generate the view engine. Engine will turn into a file
app.set("view engine", "ejs")
// have to specify where our files are going to be in this case in views directory
app.set('views', 'views')

// Here we are making requests to this page aka routing
// creating home(/) route for GET method that render home.ejs view
app.get('/', (req, res) => {
    // want to send status
    res.status(200)
    // .render(home name of our ejs file) will use the app.set('view engine', 'ejs')
    res.render('home')
})

// creating GET/searchForm route that renders searchResults.ejs view
// so here is where we actually can get the genre or rating name that needs to be visble to users
// Genre or rating id needs to be in the value attribute
app.get('/searchForm', async(req,res) =>{
    try{
        const genreSql = `
            SELECT * FROM genres;
        `

        const genreResults = await pool.query(genreSql)
        console.log(genreResults.rows)

        const ratingSql = `
            SELECT * FROM ratings;
        `

        const ratingResults = await pool.query(ratingSql)
        console.log(ratingResults.rows)


        // res.status(200)
        // rendering the ejs which express knows
        res.render('searchForm', {
            // passing in information here to 'searchForm' object
            genresObj: genreResults.rows,
            // passing in also the ratingResults
            ratingsObj: ratingResults.rows
        });
      
    }
    catch(err){
        console.log(err)
        res.status(500).send('Server Error')
    }

    
})

// creating GET/searchResults route that renders searchResults.ejs view
// We'll do the template, and the userinput array with $1, $2 etc.(only for values) that replace
// elements suc as ${userGivenTrackName}
app.get('/searchResults', async(req, res) => {
    // here within the console.log(req.query) we can see what the user inputed in the form field
    console.log(req.query)
    // we can access releaseDateFrom this way
    console.log(req.query.releaseDateFrom)

    console.log(req.query.award)
    try{
        // want to use let because we will be adding to the sql statement based on 
        // what the user provides in the form field
        let searchSql = `
            SELECT dvd_titles.title,
                    dvd_titles.award,
                    dvd_titles.release_date,
                    genres.genre,
                    ratings.rating
            FROM dvd_titles
            LEFT JOIN genres
                ON genres.genre_id = dvd_titles.genre_id
            LEFT JOIN ratings
                ON ratings.rating_id = dvd_titles.rating_id
            WHERE 1=1
        `
        // if users leave both fields empty it will search for all records bascially WHERE 1=1 tells it that
        
        // Now we want to use this searchSql in our searchResults view after we 
        // do the template and the userInputArray

        const userGivenTitle = req.query.title //got the .title from name attribute of the inside of the searchForm.ejs field. Where the action goes to /searchResults
        const userGivenGenreId = req.query.genreId
        const userGivenRatingId = req.query.ratingId
        const userGivenAwardOption = req.query.award
        const userGivenReleaseDateFrom = req.query.releaseDateFrom
        const userGivenReleaseDateTo = req.query.releaseDateTo

        // right now its going to be an empty array
        const userInputsArray = []

        // next we're doing index inorder to have an updater to keep track of the items
        // the user actually requested. Using let because we want it to be updatable.
        let index = 1;

        // now we're doing checking if using given input exists or not
        if(typeof(userGivenTitle) != 'undefined' && userGivenTitle.length >0){
            searchSql += ` AND dvd_titles.title ILIKE $` + index

            // if actually came here will increment the index
            index++
            // of course we also want to push into our userInputsArray
            userInputsArray.push(`%${userGivenTitle}%`)
        }

        if(typeof(userGivenGenreId) != 'undefined' && userGivenGenreId.length > 0){
            searchSql += ` AND dvd_titles.genre_id = $` + index
            // increment index
            index++
            // we'll push in the userGivenGenreId
            userInputsArray.push(userGivenGenreId)
        }

        if(typeof(userGivenRatingId) != 'undefined' && userGivenRatingId.length > 0){
            searchSql += ` AND dvd_titles.rating_id = $` + index
            // increment index
            index++
            // we'll push in the userGivenGenreId
            userInputsArray.push(userGivenRatingId)
        }

        if(typeof(userGivenAwardOption) != 'undefined' && userGivenAwardOption.length > 0){
            if (userGivenAwardOption == 'yes'){//will only display records that have a value in the award field AKA no null
                searchSql += ` AND dvd_titles.award IS NOT NULL`
            }else if(userGivenAwardOption == 'no'){ //user gave 'no' as an option
                searchSql += ` AND dvd_titles.award IS NULL`
            }
            // won't do anything for 'any'
            
        }

        if(typeof(userGivenReleaseDateFrom) != 'undefined' && userGivenReleaseDateFrom.length > 0){
            searchSql += ` AND dvd_titles.release_date >= $` + index
            // increment index
            index++
            // we'll push in the userGivenGenreId
            userInputsArray.push(userGivenReleaseDateFrom)
        }

         if(typeof(userGivenReleaseDateTo) != 'undefined' && userGivenReleaseDateTo.length > 0){
            searchSql += ` AND dvd_titles.release_date <= $` + index
            // increment index
            index++
            // we'll push in the userGivenGenreId
            userInputsArray.push(userGivenReleaseDateTo)
        }

        searchSql += ' LIMIT 20'

         // so now we can make the call instead of just giviing it our sql, we give it our
        //sql plus preapred statemensts that takes care of our apostrphes as well. Becuase it knows
        // its not trying to actually do any SQL injections
        const resultObject = await pool.query(searchSql, userInputsArray)

        console.log(resultObject.rows)

        // rending the searchResults view
        res.render('searchResults', {
            resultsObj: resultObject.rows
        })
    }
    catch(err){
        console.log(err)
        res.status(500).send("Sorry, could not retrieve the results.")
    }
   
})

// Want to do this at the very end. (Middleware)
// to do our 404 cause user does not get anything, its like a catch will do the get() first, and then if it
// catches it will do the .use() function
app.use((req, res) => {
    // change the status with res.status(status code)
    res.status(404)
    res.send("404 Not Found")
})

// tell computer which port to run on 
// .list(port, function)
app.listen(port, () => {
    console.log(`Server running at https://localhost:` + port)
});


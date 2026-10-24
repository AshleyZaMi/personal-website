import express from 'express';
// want to import cors to allow us to use our own API
import cors from 'cors';
import db from './db.js';

const app = express();
const port = 3000;

app.use(express.json())
app.use(cors())

// express.urlencoded() - is a middleware which will/can intercept and so we
// can use the body
app.use(express.urlencoded())

// here first endpoint want to get the /dvd_titles table
// returns: array of all dvd_title_id and title only
app.get('/dvd_titles', async(req, res) => {
    // create the sql to get all DVD titles
    try{
        const sql=`
            SELECT dvd_title_id, title FROM dvd_db.dvd_titles
        `
        // everytime you want a connection with the database use db
        const results = await db.query(sql)

        // response object
        // want to map it(the snake case postgre information) to the camelCase which is the contract
        // REST API has. map allows us to do that change row by row
        res.json(results.rows.map((row) =>{
            return{
                dvdTitleId: row.dvd_title_id,
                title: row.title
            }
        }))
    }
    catch(error){
        console.log(error)
        // typically this would be only the people who are authorized
        res.status(500).json({
            message:"Error with database"
        })
    }
})

// endpoint for '/dvd_titles/search' 
// returns: array of all dvd_title_id and title only
app.get('/dvd_titles/search', async(req, res) => {
    try{
        // typically will make these optional
        console.log(req.query)
        // cause these are from the main table so would be the id's to get the actual
        // value from the look up table
        const title = req.query.title
        const labelId = req.query.label_id
        const soundId = req.query.sound_id
        const genreId = req.query.genre_id
        const ratingId = req.query.rating_id
        const formatId = req.query.format_id

        // can query things and is optional
        let sql = `
            SELECT dvd_title_id, 
                    title
           FROM dvd_db.dvd_titles
           LEFT JOIN dvd_db.labels
                ON labels.label_id = dvd_titles.label_id
            LEFT JOIN dvd_db.sounds
                ON sounds.sound_id = dvd_titles.sound_id
            LEFT JOIN dvd_db.genres
                ON genres.genre_id = dvd_titles.genre_id
            LEFT JOIN dvd_db.ratings
                ON ratings.rating_id = dvd_titles.rating_id
            LEFT JOIN dvd_db.formats
                ON formats.format_id = dvd_titles.format_id
            WHERE 1 = 1
        `

        const userInputArray = []

        // let because we need to update the index
        let index = 1
        if(title){
            sql+=` AND dvd_titles.title ILIKE $${index}`
            // pushing the title to the userInputArray
            userInputArray.push(`%${title}%`)
            // incrementing the index
            index++
        }

         if(labelId){
            // where the id is the same
            sql+=` AND labels.label_id = $${index}`
            // pushing the labelId to the userInputArray
            userInputArray.push(labelId)
            // incrementing the index
            index++
        }

        if(soundId){
            // where the id is the same
            sql+=` AND sounds.sound_id = $${index}`
            // pushing the soundId to the userInputArray
            userInputArray.push(soundId)
            // incrementing the index
            index++
        }

        if(genreId){
            // where the id is the same
            sql+=` AND genres.genre_id = $${index}`
            // pushing the genreId to the userInputArray
            userInputArray.push(genreId)
            // incrementing the index
            index++
        }

        if(ratingId){
            // where the id is the same
            sql+=` AND ratings.rating_id = $${index}`
            // pushing the ratingId to the userInputArray
            userInputArray.push(ratingId)
            // incrementing the index
            index++
        }

        if(formatId){
            // where the id is the same
            sql+=` AND formats.format_id = $${index}`
            // pushing the formatId to the userInputArray
            userInputArray.push(formatId)
            // incrementing the index
            index++
        }

        // getting the results object
        const results = await db.query(sql, userInputArray)

        res.json(results.rows.map((row)=>{
            return{
                dvdTitleId: row.dvd_title_id,
                title: row.title
            }
        }))

    }
    catch(error){
        console.log(error)
        // typically this would be only the people who are authorized
        res.status(500).json({
            message:"Error with database"
        })
    }
})

// endpoint for '/dvd_titles/:id'
// result: get all info for a given id
app.get('/dvd_titles/:id', async(req, res) =>{
    try{
        // get all information for a given id
        const sql=`
           SELECT title,
                  release_date,
                  award,
                  label,
                  sound,
                  genre,
                  rating,
                  format
           FROM dvd_db.dvd_titles
           LEFT JOIN dvd_db.labels
                ON labels.label_id = dvd_titles.label_id
            LEFT JOIN dvd_db.sounds
                ON sounds.sound_id = dvd_titles.sound_id
            LEFT JOIN dvd_db.genres
                ON genres.genre_id = dvd_titles.genre_id
            LEFT JOIN dvd_db.ratings
                ON ratings.rating_id = dvd_titles.rating_id
            LEFT JOIN dvd_db.formats
                ON formats.format_id = dvd_titles.format_id
            WHERE dvd_title_id = $1
        `
        const inputValuesArray = [req.params.id] //this is gotten the endpoint '/dvd_titles/:id' -> ':id'
        const results = await db.query(sql, inputValuesArray)
        
        // if we do [0] we just get that object
        console.log(results.rows[0])

        // returning this response
        // res.json(results.rows[0])

        // but need to map it so will do the mapping to follow
        // the contract
        res.json(results.rows.map((row)=>{
            return{
                title: row.title,
                releaseDate: row.release_date,
                award: row.award,
                label: row.label,
                sound: row.sound,
                genre: row.genre,
                rating: row.rating,
                format: row.format
            }
            
        }))
    }
    catch(error){
        console.log(error)
        // typically this would be only the people who are authorized
        res.status(500).json({
            message:"Error with database"
        })
    }
})

// endpoint for '/dvd_titles' 
// create a new dvd record
// since it's post will be more senstive data or would be a big amount of data
// so want to save it in the body
// note: want to pass things inside of the body. Key has to be the same thing the REST API is expecting
// For this we need to add an use -> app.use(express.urlencoded())
app.post('/dvd_titles', async(req, res) =>{
    try{
        // title should not be null as defined by the properties when the database was
        // made
        // required field
        const{
            title
        } = req.body || {}

        console.log(req.body)

        console.log("BODY: ", req.body)
        
        // check if we have the required field
        if(!title){
            res.status(400).json({message:"title field is required."})
            return
        }
        
        // creating the sql to insert 
        const sql = `
            INSERT INTO dvd_db.dvd_titles(
                title,
                release_date, 
                award,
                label_id,
                sound_id,
                genre_id,
                rating_id,
                format_id
            )VALUES(
                $1,
                $2,
                $3,
                $4,
                $5,
                $6,
                $7,
                $8
            )
            RETURNING *
        `
        // so RETURNING * -> returns that row for us as well, so we can return to the user as well

        
        //contains the inital required field
        // making a array for instructions to give to the database
        const userInputArray = [
            title
        ]

        // extracting the optional fields
        const {
            releaseDate,
            award,
            labelId,
            soundId,
            genreId,
            ratingId,
            formatId
        } = req.body


        // checking if a value was given if not will assign it null to the userInputArray(that holds the instructions)
        if(!releaseDate){
            userInputArray.push(null)
        }
        else{
            // otherwise we'll push into userInputArray whatever releaseDate the user entered
            userInputArray.push(releaseDate)
        }

        // checking if a value was given if not will assign it null to the userInputArray(that holds the instructions)
        if(!award){
            userInputArray.push(null)
        }
        else{
            // otherwise we'll push into userInputArray whatever award the user entered
            userInputArray.push(award)
        }

        // checking if a value was given if not will assign it null to the userInputArray(that holds the instructions)
        if(!labelId){
            userInputArray.push(null)
        }
        else{
            // otherwise we'll push into userInputArray whatever labelId the user entered
            userInputArray.push(labelId)
        }

        // checking if a value was given if not will assign it null to the userInputArray(that holds the instructions)
        if(!soundId){
            userInputArray.push(null)
        }
        else{
            // otherwise we'll push into userInputArray whatever soundId the user entered
            userInputArray.push(soundId)
        }

        // checking if a value was given if not will assign it null to the userInputArray(that holds the instructions)
        if(!genreId){
            userInputArray.push(null)
        }
        else{
            // otherwise we'll push into userInputArray whatever genreId the user entered
            userInputArray.push(genreId)
        }

        // checking if a value was given if not will assign it null to the userInputArray(that holds the instructions)
        if(!ratingId){
            userInputArray.push(null)
        }
        else{
            // otherwise we'll push into userInputArray whatever ratingId the user entered
            userInputArray.push(ratingId)
        }

        // checking if a value was given if not will assign it null to the userInputArray(that holds the instructions)
        if(!formatId){
            userInputArray.push(null)
        }
        else{
            // otherwise we'll push into userInputArray whatever formatId the user entered
            userInputArray.push(formatId)
        }

        // try to run it
        const results = await db.query(sql, userInputArray)

        //see the results
        console.log(results)

        // want the track that is cammel case, and lets extract it
        // extract te first row, just so that we don't have to keep doing this
        const dvdObject = results.rows[0]

        // translating all the postgres snake case to cammel case(for REST api to use)
        res.status(201).json({
            title: dvdObject.title,
            releaseDate: dvdObject.release_date,
            award: dvdObject.award,
            labelId: dvdObject.label_id,
            soundId: dvdObject.sound_id,
            genreId: dvdObject.genre_id,
            ratingId: dvdObject.rating_id,
            formatId: dvdObject.format_id
        })
    }
    catch(error){
        console.log(error)
        res.status(500).json({
            message: "Server Error"
        })
    }
})

// endpoint for '/dvd_titles/:id'
// returns: update a dvd title record given an id - this version have to fill everything because if filled paritally everything else not filled will be NULL
/* app.patch('/dvd_titles/:id', async(req, res) => {
    // want to extract the id so we know what to replace
    const dvdTitleId = req.params.id //refer to it as req.params.id -> because we defined this for the path for the route
    
    // setting the possible values user can give us
    const{
        title,
        releaseDate,
        award,
        labelId,
        soundId,
        genreId,
        ratingId,
        formatId

    } = req.body

    // Create our sql
    let sql = `
        UPDATE dvd_db.dvd_titles
        SET title = $1,
            release_date = $2,
            award = $3,
            label_id = $4,
            sound_id = $5,
            genre_id = $6,
            rating_id = $7,
            format_id = $8
        WHERE dvd_title_id = $9
        RETURNING *
    `

    // right now instructions will be an empty array
    const inputValuesArray = [
        title,
        releaseDate,
        award,
        labelId,
        soundId,
        genreId,
        ratingId,
        formatId,
        dvdTitleId
    ]

    const results = await db.query(sql, inputValuesArray)

    const dvdObject = results.rows[0]

    res.json({
        title: dvdObject.title,
        releaseDate: dvdObject.release_date,
        award: dvdObject.award,
        labelId: dvdObject.label_id,
        soundId: dvdObject.sound_id,
        genreId: dvdObject.genre_id,
        ratingId: dvdObject.rating_id,
        formatId: dvdObject.format_id
    })


}) */

// endpoint for '/dvd_titles/:id'
// returns: update a dvd title record given an id - this version is dynamic first make the arrays
// for the fields for SET, and then input the actual values in the inputValuesArray. Then seperate those with
// commas using join and then just lastly use the id
app.patch('/dvd_titles/:id', async(req, res) => {
    try{
        // want to extract the id so we know what to replace
        const dvdTitleId = req.params.id //refer to it as req.params.id -> because we defined this for the path for the route
        
        let fieldsArray = []
        let inputValuesArray = []
        let index = 1

        // setting the possible values user can give us
        const{
            title,
            releaseDate,
            award,
            labelId,
            soundId,
            genreId,
            ratingId,
            formatId
        } = req.body

        // making arrays first
        if (title !== undefined){
            fieldsArray.push(`title = $${index}`)
            index++
            inputValuesArray.push(title)
            
        }

        if (releaseDate !== undefined){
            fieldsArray.push(`release_date = $${index}`)
            index++
            inputValuesArray.push(releaseDate) 
        }

        if (award !== undefined){
            fieldsArray.push(`award = $${index}`)
            index++
            inputValuesArray.push(award) 
        }

        if (labelId !== undefined){
            fieldsArray.push(`label_id = $${index}`)
            index++
            inputValuesArray.push(labelId) 
        }

        if (soundId !== undefined){
            fieldsArray.push(`sound_id = $${index}`)
            index++
            inputValuesArray.push(soundId) 
        }

        if (genreId !== undefined){
            fieldsArray.push(`genre_id = $${index}`)
            index++
            inputValuesArray.push(genreId) 
        }

        if (ratingId !== undefined){
            fieldsArray.push(`rating_id = $${index}`)
            index++
            inputValuesArray.push(ratingId) 
        }

        if (formatId !== undefined){
            fieldsArray.push(`format_id = $${index}`)
            index++
            inputValuesArray.push(formatId) 
        }

        // Create our sql
        let sql = `
            UPDATE dvd_db.dvd_titles
            SET ${fieldsArray.join(', ')}
            WHERE dvd_title_id = $${index}
            RETURNING *
        `

        // add id last
        inputValuesArray.push(dvdTitleId)
        
        const results = await db.query(sql, inputValuesArray)

        const dvdObject = results.rows[0]

        res.json({
            title: dvdObject.title,
            releaseDate: dvdObject.release_date,
            award: dvdObject.award,
            labelId: dvdObject.label_id,
            soundId: dvdObject.sound_id,
            genreId: dvdObject.genre_id,
            ratingId: dvdObject.rating_id,
            formatId: dvdObject.format_id
        })
    }
    catch(error){
        console.log(error)
        res.status(500).json({
            message: "Server Error"
        })
    }

})

// endpoint for '/dvd_titles/:id'
// returns: delete a dvd record given id(this will be dvd_title_id) 
app.delete('/dvd_titles/:id', async(req, res) => {
    try{
        const dvdTitleId = req.params.id

        // Create our sql
        //returning everything with RETURNING *
        const sql = `
            DELETE FROM dvd_db.dvd_titles
            WHERE dvd_title_id = $1
            RETURNING *
        `

        // need an array
        const userInputArray = [dvdTitleId]

        // getting the results
        const results = await db.query(sql, userInputArray)

        // getting a single dvd 
        const dvdObject = results.rows[0]

        // returning the response(which would be like popping an object
        // from the stack)
        res.json({
            title: dvdObject.title,
            releaseDate: dvdObject.release_date,
            award: dvdObject.award,
            labelId: dvdObject.label_id,
            soundId: dvdObject.sound_id,
            genreId: dvdObject.genre_id,
            ratingId: dvdObject.rating_id,
            formatId: dvdObject.format_id
        })

    }
    catch(error){
        console.log(error)
        res.status(500).json({
            message: "Server Error"
        })
    }
})

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`)
})
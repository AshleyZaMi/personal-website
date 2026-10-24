// this is where my logic goes, replacing index.js code
import bcrypt from 'bcryptjs'
import db from '../db.js'

// Login 
export const login = async (req, res) => {
    try{
        const {username, password} = req.body

        // try to login, so all we have to do is check the dataase and see if the username and 
        // password matches
        const sql = `
            SELECT * FROM users
            WHERE username= $1
        `
        const userInputs = [username]
        const result = await db.query(sql, userInputs)

        // if we get nothing means its not there
        if (result.rows.length === 0){
            return res.status(400).json({
                message: "Invalid username"
            })
        }

        // after we know the user is valid so performing next operations
        const validUser = result.rows[0]


        // 👇 PUT DEBUG LOGS RIGHT HERE
        console.log("DB USER:", validUser)
        console.log("INPUT PASSWORD:", password)
        console.log("DB HASH:", validUser.password)

        const match = await bcrypt.compare(password, validUser.password)

        if (!match){
            return res.status(400).json({
                message:"Invalid password"
            })
        }

        // save new user session
        // Beforre we redirect we want to start our session
        // And we can create as many session objects we want
        req.session.variableName = 'variableValue'
        req.session.user = {
            id: validUser.user_id,
            username:validUser.username,
            email:validUser.email
        }

        return res.json({
            message: "Login successful",
            user: req.session.user //returning user to frontend
        })
    }catch(err){
        console.log(err)
        res.status(500).json({
            message: "Server error"
        })
    }
} 

// Register
export const register = async (req, res) => {
    try{
        const {username, email, password} = req.body

        // now will validate
        const sqlAlreadyRegisteredUsers = `
            SELECT * FROM users 
            WHERE username = $1
                OR email = $2
        `
        const givenUserInputs = [username, email]
        // check if the username and email already exist
        const check = await db.query(sqlAlreadyRegisteredUsers, givenUserInputs)

        // checking
        if (check.rows.length > 0){
            return res.status(400).json({
                message: "User already exists"
            })
        }

        // By this point it was already checked that username and email are unique
        // hashing the password
        const hashedPass = await bcrypt.hash(password, 10)

        // now will register the user
        const sqlRegisterUser = `
            INSERT INTO users
            (username, email, password)
            VALUES
            ($1, $2, $3);
        `

        // passiing in the user given values
        const givenUserFinalInputs = [username, email, hashedPass]

        const results = await db.query(sqlRegisterUser, givenUserFinalInputs)

        // just telling the frontend it worked, frontend handles the redirect
        res.status(201).json({
            message: "User created successfully"
        })

    } catch(err){
        console.log(err)
        // res.status(500).json({
        //     message: "Server error"
        // })
        // res.status(500).json({
        //     message: err.message,
        //     username
        // })
        return res.status(500).json({
            message: err.message,
            detail: err.detail
        })
    }
}

// Logout
export const logout = (req, res) => {
    req.session.destroy(() => {
        res.json({message: "Logged out"})
    })
}
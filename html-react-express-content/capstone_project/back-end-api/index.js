// startup the server with express for RESTAPI
import express from 'express'
import session from 'express-session'
import cors from 'cors'

import authRoutes from './routes/authRoutes.js'
import sentRoutes from './routes/sentRoutes.js'
import quoteRoutes from './routes/quoteRoutes.js'
import taskRoutes from './routes/taskRoutes.js'
import pomodoroRoutes from './routes/pomodoroRoutes.js'
import blogRoutes from './routes/blogRoutes.js'

const app = express()
const port = 3000

//Body parser middleware first
app.use(express.urlencoded({extended: false}))
app.use(express.json())

// other middleware(cors, auth)
app.use(cors({
    origin: process.env.FRONTEND_URL, //the React frontend
    credentials: true
}))

app.use(session({
    secret: process.env.SESSION_SECRET,//used to generate random string set by us not random people
    resave: false,
    saveUninitialized: false,
    cookie: {
        secure: true,      //required for HTTPS on Render
        sameSite: 'none'   //required for cross-origin requests
    }
}))

// mounting routes
app.use('/', authRoutes)
app.use('/api', sentRoutes)
app.use('/api', quoteRoutes)
app.use('/api/tasks', taskRoutes)
app.use('/api', pomodoroRoutes)
app.use('/api', blogRoutes)


app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`)
})
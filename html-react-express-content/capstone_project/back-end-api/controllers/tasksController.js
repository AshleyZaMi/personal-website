import db from '../db.js'

// Get all tasks
export const getTasks = async(req,res) =>{
    try{
        // check if user is logged in first
        if (!req.session.user) {
            return res.status(401).json({ error: 'Not logged in' })
        }
        // getting the current user_id
        const user_id = req.session.user.id

        // making sql statement to get all tasks information
        const sql = `
        SELECT task_id, task_name, where_field, time_to_spend, completed
        FROM tasks 
        WHERE user_id = $1
        ORDER BY task_id ASC
        `
        // saving the results gotten from db
        const result = await db.query(sql, [user_id])

        // normalizing keys for frontend
        const formatted = result.rows.map(row => ({
            id: row.task_id,
            taskName: row.task_name,
            where: row.where_field,
            timeToSpend: row.time_to_spend,
            completed: row.completed
        }))

        // sending the formatted json object to front end
        res.json(formatted)

    }catch(err){
        res.status(500).json({error: err.message})
    }
}

// creating task
export const createTask = async (req, res) => {
    try{
        // check if user is logged in first
        if (!req.session.user) {
            return res.status(401).json({ error: 'Not logged in' })
        }

        const {taskName, where, timeToSpend} = req.body
        console.log("BODY:", req.body)

        if (!taskName || !where || !timeToSpend) {
            return res.status(400).json({ error: "Missing fields" })
        }

        if (isNaN(timeToSpend)) {
            return res.status(400).json({ error: "timeToSpend must be a number" })
        }
        
        // getting the user_id to put in table to keep track who that task belongs to
        const user_id = req.session.user.id

        const insertSql = `
            INSERT INTO tasks (user_id, task_name, where_field, time_to_spend)
            VALUES ($1, $2, $3, $4)
            RETURNING *
        `
        const userInputs = [user_id, taskName, where, timeToSpend]

        const result = await db.query(insertSql, userInputs)

        // getting row to work easily with it
        const row = result.rows[0]
        console.log(row)
        const formatted = {
            id: row.task_id,
            taskName: row.task_name,
            where: row.where_field,
            timeToSpend: row.time_to_spend,
            completed: row.completed
        }

        // sending the formatted json object to front end
        res.json(formatted)
    }catch(err){
        res.status(500).json({error: err.message})
    }
}

// update task
export const updateTask = async (req, res) => {
    try{
        // check if user is logged in first
        if (!req.session.user) {
            return res.status(401).json({ error: 'Not logged in' })
        }

        const {id} = req.params
        const {taskName, where, timeToSpend} = req.body

        // only update the tasks associated to that user
        const user_id = req.session.user.id

        const updateSql = `
            UPDATE tasks
            SET task_name = $1,
                where_field = $2,
                time_to_spend = $3
            WHERE task_id = $4
            AND user_id = $5
            RETURNING *
        `

        const userInputs = [taskName, where, timeToSpend, id, user_id]

        const result = await db.query(updateSql, userInputs)

        // using row for more simpler mapping for normalization so the
        // front end could understand
        const row = result.rows[0]
        console.log("PARAM ID:", req.params.id)
        res.json({
            id: row.task_id,
            taskName: row.task_name,
            where: row.where_field,
            timeToSpend: row.time_to_spend,
            completed: row.completed
        })
    }catch(err){
        res.status(500).json({error: err.message})
    }
}

// Delete task
export const deleteTask = async (req,res) => {
    try{
        // check if user is logged in first
        if (!req.session.user) {
            return res.status(401).json({ error: 'Not logged in' })
        }

        const {id} = req.params

        // making sure the task is with that user_id
        const user_id = req.session.user.id
        const deleteSql = `
            DELETE FROM tasks WHERE task_id = $1 AND user_id = $2
        `

        const userInputs = [id, user_id]

        const result = await db.query(deleteSql, userInputs)

        res.json({message: "Task deleted"})
    }catch(err){
        res.status(500).json({error: err.message})
    }
}
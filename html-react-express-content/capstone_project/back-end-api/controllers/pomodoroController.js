import db from '../db.js'

export const createSession = async (req, res) => {

    try {

        const {task_id, session_duration} = req.body

        // error checking
        if (!task_id || !session_duration) {
            return res.status(400).json({ message: 'task_id and session_duration are required' })
        }

        const createSessionSql = `
            INSERT INTO pomodoro_sessions
            (task_id, session_duration)
            VALUES
            ($1, $2)
            RETURNING *
        `

        const userInputs = [
            task_id,
            session_duration
        ]

        const result = await db.query(createSessionSql, userInputs)

         // getting row to work easily with it
        const row = result.rows[0]
        console.log(row)
        const formatted = {
            id: row.session_id,
            taskId: row.task_id,
            sessionDuration: row.session_duration,
            completedAt: row.completed_at
        }
        // sending the formatted json object to front end
        res.json(formatted)

    } catch(err) {

        console.log(err)

        res.status(500).json({
            message: 'Server error'
        })
    }
}
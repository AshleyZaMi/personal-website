import express from 'express'

import{
    getTasks,
    createTask,
    updateTask,
    deleteTask
} from '../controllers/tasksController.js'

const router = express.Router()

// here are the routes I have for CRUD functuanality
router.get('/', getTasks)
router.post('/', createTask)
router.put('/:id', updateTask)
router.delete('/:id', deleteTask)

export default router;
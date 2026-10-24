import { useState, useEffect } from 'react'
import '../styles/TaskManager.css'
import SectionName from '../components/SectionName.jsx'
import Task from '../components/Task.jsx'
import TaskAdd from './TaskAdd.jsx'

// will have a live one now so will replace to this path
const API_BASE_URL = import.meta.env.VITE_API_URL

export default function TaskManager(props) {
    //setting the inital list for task objects
    const [taskObjectsList, setTaskObjectsList] = useState([]);
    // opening editing mode
    const [editingTaskId, setEditingTaskId] = useState(null)
    const [editForm, setEditForm] = useState({})


    // Loading data given from RESTAPI at the begining only once
    useEffect(() => {
        fetch(API_BASE_URL + '/api/tasks', {
            credentials: 'include'
        })
            .then(res => res.json())
            .then(data =>{
                if (Array.isArray(data)) {
                    setTaskObjectsList(data)
                } else {
                    console.error("Backend error:", data)
                    setTaskObjectsList([]) // safe fallback
                }
                })
    }, [])

    // want to have a handler to take care of the TaskAdd child's given feedback.

    // for create funtionality
    // Making a function to handle brand new products added
    const handleNewTaskAddCompAddition = async(taskObject) => {
        
        try{
            // awaiting the response from RESTAPI
            const res = await fetch(API_BASE_URL + '/api/tasks', {
                method:'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(taskObject),
                credentials: 'include'
            })

            // fixing the error
            if (!res.ok){
                const err = await res.json()
                console.error('Failed to create task:', err)
                return //stop here, don't update the list
            }
            // getting the new task object
            const newTaskObject = await res.json()


            //updating the product object list with the new object included 
            setTaskObjectsList((prevObjects) =>{
                const newTaskObjectsList = [
                    ...prevObjects,
                    newTaskObject
                ]

                return newTaskObjectsList
            })
        }catch(err){
            console.err('Netrowk error:', err)
        }
    }

    // For delete functionality
    // Child(Task) wants to delete an object.
    const handleTaskCompLevelDelete = async(id) => {
        try{
            // delete sends an empty body
            const res = await fetch(`${API_BASE_URL}/api/tasks/${id}`, {
                method: 'DELETE',
                credentials: 'include'
            })

            // fixing the error
            if (!res.ok){
                const err = await res.json()
                console.error('Failed to create task:', err)
                return //stop here, don't update the list
            }

            
            // here we want to filter, will return true if not the same
            setTaskObjectsList((prevObjects) => {
                const newTaskObjectsFilteredList = prevObjects.filter((currentTaskObject) =>{
                    // only keeping the objects that are not the task object we want to delete
                    if (currentTaskObject.id !== id){//if both are not the same
                        return true//so that we copy it to our new Task Objects array
                    }
                })

                return newTaskObjectsFilteredList
            })
        }catch(err){
            console.error('Network error:', err)
        }
    }

    // updating the task
    const handleUpdateTask = async(updatedTask) => {
        try{
            if (!updatedTask.id) {
                console.error("Missing task ID:", updatedTask)
                return
            }
            // getting restAPI repsonse
            const res = await fetch(
                `${API_BASE_URL}/api/tasks/${updatedTask.id}`,
                {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(updatedTask),
                    credentials:'include'
                }
            )

            // fixing the error
            if (!res.ok){
                const err = await res.json()
                console.error('Failed to create task:', err)
                return //stop here, don't update the list
            }

            // getting the updated object
            const savedObject = await res.json()

            setTaskObjectsList((prevObjects)=> {
                const updatedObjectsList = prevObjects.map((currentTaskObject) =>{
                    if (currentTaskObject.id === savedObject.id){
                        return savedObject;//returning this object
                    }
                    // else retuning the current task
                    return currentTaskObject
                })
                // returning the updated objects list
                return updatedObjectsList
            })
                
            setEditingTaskId(null)//closing the editor after saving
        }catch(err){
            console.error('Network error:', err)
        }
    }


    // Open edit mode
    // const handleEditClick = (taskObject) => {
    //     setEditingTask(taskObject.id)

    //     setEditForm({
    //         taskName:taskObject.taskName,
    //         where: taskObject.where,
    //         timeToSpend: taskObject.timeToSpend
    //     })
    // }


    return (
        <>
            <div id='task-manager-wrapper'>
                <SectionName
                    sectionName='Task Manager'
                    className='section-name-update'
                />

                {/* will need like a for loop here in order
            to update dynamically */}
                {/* Here making sure to allow the child to add a brand new task via a method the parent gives access to the child*/}
                <div id='new-task-creator-holder'>
                    <TaskAdd
                        parentAddNewTaskMethodHandler={handleNewTaskAddCompAddition}
                    />
                </div>


                <div id='tasks-holder'>

                    {/* adding tasks */}
                    {/* going into javascript mode, for the map to loop over the object array of products */}

                    {taskObjectsList.map((taskObject) => {
                        const isEditing = editingTaskId === taskObject.id
                        return (
                            <Task
                                key={taskObject.id}
                                // this will help us keep track of the current object
                                currTaskObject={taskObject}

                                isEditing={isEditing}
                                setEditingTaskId={setEditingTaskId}

                                editForm={editForm}
                                setEditForm={setEditForm}

                                // onEdit={handleEditClick}
                                onSave={handleUpdateTask}
                                onDelete={handleTaskCompLevelDelete}
                                // Passing the information to the child from the current parent level,
                                // so the the child has access to key/property/attribute -> parentMethodHandler, so the child can use
                                // handleAddProduct, and hadnleProductRemove(both to notify the child level updates it wants the parent 
                                //to take care of)
                                // parentTaskCompLevelDeleteMethodHandler={handleTaskCompLevelDelete}
                            />
                        )

                    })}
                    {/* <Task
                        taskId={1}
                        taskName='Reviewing flashcard'
                        taskLocation='Link or just text'
                    />
                    <Task
                        taskId={1}
                        taskName='Reviewing flashcard'
                        taskLocation='Link or just text'
                    /> */}

                </div>

            </div>

        </>
    )
}
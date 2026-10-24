import '../styles/PomodormoSection.css'
import SectionName from '../components/SectionName.jsx'
import PinkField from '../components/PinkField.jsx'
import Button from '../components/Button.jsx'

import { useTimer } from 'react-timer-hook'
import { useState, useEffect } from 'react'

// will have a live one now so will replace to this path
const API_BASE_URL = import.meta.env.VITE_API_URL

// outside compoenent so its not created on every render
// helper to get expiry time from seconds
const getExpiryTime = (secs) => {
    const time = new Date()
    time.setSeconds(time.getSeconds() + secs)
    return time
}
export default function PomodormoSection(props) {
    // selected task making it dynamic
    const [selectedTask, setSelectedTask] = useState('')
    // this is to fetch tasks to update the selections available from db
    const [tasks, setTasks] = useState([])

    // pomodormo duration(25 mins)
    const [duration, setDuration] = useState(1500)

    // here this prevents auto-saving on the first render
    const [hasStarted, setHasStarted] = useState(false)
    const [expiryTime, setExpiryTime] = useState(null)
    
    // timer set up
    // these are to use the things from the library from node for the timer
    const {
        seconds,
        minutes,
        hours,
        start,
        pause,
        restart
    } = useTimer({
        expiryTimestamp: getExpiryTime(duration),
        autoStart: false,
        onExpire: () => console.log('timer expired!')
    })

    // fetching tasks from RestAPI
    useEffect(() => {
        fetch(API_BASE_URL + '/api/tasks', {
            credentials: 'include'
        })
            .then(res => res.json())
            .then(data => {
                // only if its actually an array
                if (Array.isArray(data)) {
                    setTasks(data)
                    console.log('tasks fetched:', data)
                } else {
                    setTasks([])
                }
            })
            .catch(err => console.log(err))
    }, [])

    // check for saved timer on mount(restart possibility exists here)
    useEffect(() => {
        const savedExpiry = localStorage.getItem('pomodoroExpiry')
        const savedTask = localStorage.getItem('pomodoroTask')
        if (savedExpiry && savedTask) {
            const expiry = new Date(savedExpiry)
            if (expiry > new Date()) {
                // timer still going, resume it
                setSelectedTask(Number(savedTask))
                setExpiryTime(expiry)
                restart(expiry, true)
                setHasStarted(true)
            } else {
                // timer expired while away, clean up
                localStorage.removeItem('pomodoroExpiry')
                localStorage.removeItem('pomodoroTask')
            }
        }
    }, [])


    //calculate display manually to handle 60+ minutes correctly
    const totalMinutes = hours * 60 + minutes
    const displayMinutes = String(totalMinutes).padStart(2, '0')
    const displaySeconds = String(seconds).padStart(2, '0')


    // when they select a task, update duration too
    const handleTaskChange = (event) => {
        const selectedId = Number(event.target.value)
        setSelectedTask(selectedId)

        // find the task they picked and use its timeToSpend
        const task = tasks.find(task => task.id == selectedId)

        if (task) {
            const newDuration = task.timeToSpend * 60 //Convert mins to seconds
            const newExpiry = getExpiryTime(newDuration)
            console.log('timeToSpend:', task.timeToSpend)  // is it 60?
            console.log('newDuration in seconds:', newDuration)  // should be 3600
            console.log('expiryTime:', newExpiry)
            setDuration(newDuration)
            // saving expiry time
            setExpiryTime(newExpiry)
            restart(newExpiry, false) //false = don't autostart
            setHasStarted(false)
        }
    }

    //resume timer correctly when coming baxk to tab
    useEffect(() => {
        const handleVisibilityChange = () => {
            if (document.visibilityState === 'visible' && hasStarted && expiryTime) {
                restart(expiryTime, true) // resume from saved expiry
            }
        }
        document.addEventListener('visibilitychange', handleVisibilityChange)
        return () => {
            document.removeEventListener('visibilitychange', handleVisibilityChange)
        }
    }, [hasStarted, expiryTime])

    // save completed session
    // function to save the session:
    const saveSession = async (duration) => {
        // don't save if no task selected
        if (!selectedTask) return
        try {
            const response = await fetch(API_BASE_URL+"/api/sessions", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                credentials: "include",
                body: JSON.stringify({
                    task_id: selectedTask,
                    session_duration: duration
                })
            })
            const result = await response.json()

            console.log(result)
        } catch (err) {
            console.log(err)
        }
    }

    // when timer finishes
    useEffect(() => {
        if (hasStarted &&
            minutes === 0 && seconds === 0 && hours === 0) {
            saveSession(duration)
            setHasStarted(false)
            // clean up localStorage
            localStorage.removeItem('pomodoroExpiry')
            localStorage.removeItem('pomodoroTask')
        }
    }, [minutes, seconds, hours])//removing others so it only watches the timer

    return (
        <>
            <div id='pomo-wrapper'>
                <SectionName
                    sectionName='Pomodormo'
                    className='pomo-section-name-update'
                />

                <PinkField
                    text={`${displayMinutes}:${displaySeconds}`}
                    className='pomo-pink-field-update'
                />

                {/* dropdown with the tasks that are in the task manager */}
                {/* making it dynamic */}
                <div id="task-dropdown">
                    <label htmlFor='task-type'>Which Task? </label>
                    <select
                        id='task-type'
                        value={selectedTask}
                        onChange={handleTaskChange}
                    >
                        <option value='' disabled>Select a task</option>
                        {tasks.map(task => (
                            <option key={task.id} value={task.id}>
                                Task {task.id}: {task.taskName} ({task.timeToSpend} mins)
                            </option>
                        ))}

                    </select>
                </div>

                {/* buttons */}
                {/* start and stop button */}
                <div id='button-manager'>
                    <Button
                        buttonText='Start'
                        className='start-button'
                        // passing start
                        onClick={() => {
                            const newExpiry = getExpiryTime(duration)
                            setHasStarted(true)
                            // save expiry on start
                            setExpiryTime(newExpiry)
                            // save to local storage to navigate across browsers
                            localStorage.setItem('pomodoroExpiry', newExpiry.toISOString())
                            localStorage.setItem('pomodoroTask', String(selectedTask))
                            start()
                        }}
                    />


                    <Button
                        buttonText='Stop'
                        className='stop-button'
                        onClick={pause}
                    />


                </div>

            </div>
        </>
    )
}
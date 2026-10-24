import { useState } from 'react';
import '../styles/TaskAdd.css';

// TaskAdd will be a child of TaskManager
export default function TaskAdd(props) {
    const [taskNameInput, setTaskName] = useState('');
    const [whereInput, setWhere] = useState('');
    const [timeToSpendInput, setTimeToSpend] = useState('');

    // The child is taking care in this method to inform the parent about the changes
    // done to the form field on a child scope. When form is submitted and the method is used in OnSubmit
    function childHandleSubmitFormActions(e) {
       if(e){
        e.preventDefault();
       } 

       if (!taskNameInput || !whereInput || !timeToSpendInput) {
            return; // block empty submission
        }
        // creating a new task object
        const newTaskObject = {
            taskName: taskNameInput,
            where: whereInput,
            timeToSpend: Number(timeToSpendInput)
        }

        // lets go ahead and pass the new object into the parent method handler
        props.parentAddNewTaskMethodHandler(newTaskObject)

        // Setting the values to empty so the user could type in new things everytime
        setTaskName('');
        setWhere('');
        setTimeToSpend('');//this shows the default which is Select an option
    }

    // defining isValidURL for where
    const isValidUrl = (str) => {
        try {
            // getting the url and if it is sucessful then returning true
            const url = new URL(str.startsWith('http') ? str : `https://${str}`);
            return /^[\w-]+\.[a-z]{2,}$/i.test(url.hostname);;
        } catch {
            // otherwise its a string
            return false;
        }
    }

    // defining getUrl
    const getUrl = (str) =>
        str.startsWith('http') ? str : `https://${str}`;


    return (
        <>
            {/* <div className={`task-container ${props.className || ""}`}>
            <p>Task {props.taskId}: {props.taskName}</p>
            <p>Where: {props.taskLocation}</p>
            <p>Time to spend:{props.taskTimeAmount}</p>
        </div> */}

            <div id='add-new-task-wrapper'>

                {/* creating a form to handle the adding objects information gathered from user */}
                {/* child(this file) will handle the submitting form actions */}
                {/* onSubmit will handle if people press enter to submit */}
                <form>
                    {/* using the task-container class from Task whcish will help it get its structure */}
                    <div className='task-container'>
                        <div id='add-button'>
                            {/* this is the (x) delete button */}
                            <span className='material-symbols-outlined'
                                // here we want to delete the task object on click from Task component scope, child give their action 
                                // to the parent via onClick function
                                onClick={() => {
                                    // Calling the function so it can be in charge of adding the new task object to list
                                    childHandleSubmitFormActions()
                                }}
                            >
                                add_circle
                            </span>
                        </div>
                        {/* This is for 'Task' */}
                        <div className='task-row'>
                            <label htmlFor='taskName'>Task: </label>
                            <input
                                id='taskName'
                                type='text'
                                className='form-control'
                                // onChange lets us update our state, it is vound to the UI
                                onChange={(e) => {
                                    setTaskName(e.target.value)
                                }}
                                value={taskNameInput}//everytime taskNameInput changes our react will change it 
                            ></input>
                        </div>

                        <div className='task-row'>
                            {/* this label is connected to the id */}
                            <label htmlFor='where'>Where: </label>
                            {/* checkinf it url is valid */}

                            <input
                                // id so that we can use in css/id is elements name in page
                                id='where'
                                className='form-control'
                                onChange={(e) => {
                                    setWhere(e.target.value)
                                }}
                                value={whereInput}
                            >
                            </input>

                            {/* rendering the results outside of input */}
                            {/* checking if url is valid */}
                            {whereInput && isValidUrl(whereInput) && (
                                <a
                                    href={getUrl(whereInput)}
                                    target="_blank"
                                    rel="noreferrer"
                                >
                                    Open Link
                                </a>
                            )}

                        </div>

                        <div className='task-row'>
                            <label htmlFor='timeToSpend'>Time: </label>
                            <select
                                id='timeToSpend'
                                className='form-control'

                                // setttin the value here
                                onChange={(e) => {
                                    setTimeToSpend(e.target.value)
                                }}
                                // value is the value it shows
                                value={timeToSpendInput}
                            >
                                {/* first option is not selectable */}
                                <option value="" disabled>Select an option</option>
                                <option value="5">5 minutes</option>
                                <option value="10">10 minutes</option>
                                <option value='20'>20 minutes</option>
                                <option value='30'>30 minutes</option>
                                <option value='40'>40 minutes</option>
                                <option value='50'>50 minutes</option>
                                <option value='60'>60 minutes</option>

                            </select>
                        </div>

                    </div>
                </form>

            </div>
        </>
    )
}
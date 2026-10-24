import '../styles/Task.css'

export default function Task(props) {
    // getting the current task object
    const task = props.currTaskObject
    const isEditing = props.isEditing

    if(!task?.id) return null;

    // checking if its in edit mode
    if (isEditing) {
        return (
            // editing the taskName
            <div className='task-container'>
                <div className='task-row'>
                    <label htmlFor='taskName'>Task: </label>
                    <input
                        id='taskName'
                        className='form-control'
                        type='text'
                        value={props.editForm.taskName ?? ""}
                        onChange={(event) => {
                            props.setEditForm({
                                ...props.editForm,
                                taskName: event.target.value
                            })
                        }} />
                </div>

                {/* editing the where */}
                <div className='task-row'>
                    {/* this label is connected to the id */}
                    <label htmlFor='where'>Where: </label>
                    {/* checkinf it url is valid */}

                    <input
                        id='where'
                        className='form-control'
                        value={props.editForm.where || ""}
                        onChange={(event) => {
                            props.setEditForm({
                                ...props.editForm,
                                where: event.target.value
                            })
                        }} />
                </div>

                {/* editing the timeToSpend */}
                 <div className='task-row'>
                    <label htmlFor='timeToSpend'>Time: </label>
                    <select
                        id='timeToSpend'
                        className='form-control'
                        // value is the value it shows
                        value={props.editForm.timeToSpend}
                        // setttin the value here
                        onChange={(event) => {
                            props.setEditForm({
                                ...props.editForm,
                                timeToSpend: event.target.value
                            })
                        }}
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

                <div id='buttons-container'>
                    <button onClick={() => props.onSave(props.editForm)}>
                        Save
                    </button>

                    <button onClick={() => {
                        props.setEditingTaskId(null)
                        props.setEditForm({})
                    }}>
                        Cancel
                    </button>
                </div>
            </div>
        )
    }
    // regular return
    return (
        <>
            <div className={`task-container ${props.className || ""}`}>
                <div id='buttons-container'>
                    {/* adding a [edit] button */}
                    <div id='update-button'>
                        <span
                            className="material-symbols-outlined"
                            onClick={() => {
                                // setting these from TaskManger such that passing values
                                props.setEditingTaskId(task.id);
                                props.setEditForm({
                                id: task.id,
                                taskName: task.taskName,
                                where: task.where,
                                timeToSpend: task.timeToSpend
                                }
                                )//sending a copy to avoid accidental mutation bugs
                            }}
                        >
                            edit_square
                        </span>
                    </div>
                    <div id='remove-button'>
                        {/* this is the (x) delete button */}
                        <span className='material-symbols-outlined'
                            // here we want to delete the task object on click from Task component scope, child give their action 
                            // to the parent via onClick function
                            onClick={() => {
                                // giving the parent the whole taskObject via .currTaskObject that is defined as an attribute in the TaskManager.jsx(parent)
                                // when they're showing the child component tasks via a loop
                                props.onDelete(task.id)
                            }}
                        >
                            cancel
                        </span>

                    </div>
                </div>

                <p>Task {task.id}: {task.taskName}</p>
                <p>Where: {task.where}</p>
                <p>Time to spend: {task.timeToSpend} minutes</p>
            </div>


        </>
    )
}
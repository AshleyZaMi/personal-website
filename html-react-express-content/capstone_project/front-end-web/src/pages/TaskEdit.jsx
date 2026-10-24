import {useState, useEffect} from 'react'
import '../styles/TaskAdd.css'

export default function TaskEdit(props){
    const [taskNameInput, setTaskName] = useState('')
    const [whereInput, setWhere] = useState('')
    const [timeToSpendInput, setTimeToSpend] = useState('')

    // load existing task into form when component opens
    useEffect(() => {
        if (props.task) {
            setTaskName(props.task.taskName);
            setWhere(props.task.where);
            setTimeToSpend(props.task.timeToSpend);
        }
    }, [props.task]);

    function handleSave(e) {
        if (e) e.preventDefault();

        const updatedTask = {
            ...props.task,
            taskName: taskNameInput,
            where: whereInput,
            timeToSpend: timeToSpendInput
        };

        props.onUpdateTask(updatedTask);
    }

    return (
        <div id="edit-task-wrapper">

            <form onSubmit={handleSave}>
                <div className="task-container">

                    {/* SAVE BUTTON (edit icon) */}
                    <div id="edit-button">
                        <span
                            className="material-symbols-outlined"
                            onClick={handleSave}
                        >
                            edit_square
                        </span>
                    </div>

                    {/* TASK NAME */}
                    <div className="task-row">
                        <label>Task:</label>
                        <input
                            value={taskNameInput}
                            onChange={(e) => setTaskName(e.target.value)}
                        />
                    </div>

                    {/* WHERE */}
                    <div className="task-row">
                        <label>Where:</label>
                        <input
                            value={whereInput}
                            onChange={(e) => setWhere(e.target.value)}
                        />
                    </div>

                    {/* TIME */}
                    <div className="task-row">
                        <label>Time:</label>
                        <select
                            value={timeToSpendInput}
                            onChange={(e) => setTimeToSpend(e.target.value)}
                        >
                            <option value="5">5 min</option>
                            <option value="10">10 min</option>
                            <option value="20">20 min</option>
                            <option value="30">30 min</option>
                            <option value="60">60 min</option>
                        </select>
                    </div>

                </div>
            </form>

        </div>
    );
}
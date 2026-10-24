import './Task.css'

export default function Task(props){
    return(
        <>
        <div className={`task-container ${props.className || ""}`}>
            <p>Task {props.taskId}: {props.taskName}</p>
            <p>Where: {props.taskLocation}</p>
            <p>Time to spend:{props.taskTimeAmount}</p>
        </div>
        </>
    )
}
import './PomodormoSection.css'
import SectionName from './SectionName.jsx'
import PinkField from './PinkField.jsx'
import Button from './Button.jsx'

export default function PomodormoSection(props){
    return(
        <>
        <div id='pomo-wrapper'>
            <SectionName
            sectionName='Pomodormo'
            className='pomo-section-name-update'
           
            ></SectionName>
            <PinkField
            text="00:05:00"
            className='pomo-pink-field-update'
            ></PinkField>
            {/* dropdown with the tasks that are in the task manager */}
            <div id="task-dropdown">
                <label htmlFor='task-type'>Which Task? </label>
                <select id='task-type'>
                    <option value='task1'>Task 1: Reviewing Flashcard</option>
                    <option value='task2'>Task 2: Watching Videos</option>
                    <option value='task3'>Task 3: Grammar Review</option>
                    <option value='task4'>Task 4: Write a blog post</option>
                    <option value='task5'>Task 5: Do some sentence analysis</option>
                </select>
            </div>
            {/* start and stop button */}
            <div id='button-manager'>
                <Button
                buttonText='Start'
                className='start-button'
                >
                </Button>

                <Button
                buttonText='Stop'
                className='stop-button'>
                </Button>
            </div>

        </div>
        </>
    )
}
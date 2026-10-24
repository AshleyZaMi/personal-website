import './TaskManager.css'
import SectionName from './SectionName.jsx'
import Task from './Task.jsx'

export default function TaskManager(props){
    return(
        <>
        <div id='task-manager-wrapper'>
            <SectionName
            sectionName='Task Manager'
            className='section-name-update'
            />
            {/* will need like a for loop here in order
            to update dynamically */}
            <div id='tasks-holder'>
                <Task
                taskId = {1}
                taskName= 'Reviewing flashcard'
                taskLocation= 'Link or just text'
                />
                <Task
                taskId = {2}
                taskName= 'Watching Videos'
                taskLocation= 'Link or just text'
                />
                <Task
                taskId = {3}
                taskName= 'Grammar Review'
                taskLocation= 'Link or just text'
                />
                <Task
                taskId= {4}
                taskName= 'Write a blog post'
                taskLocation= 'Link or just text'
                />
                <Task
                taskId= {5}
                taskName= 'Do some sentence analysis'
                taskLocation= 'Link or just text'
                />
            </div>

        </div>
        </>
    )
}
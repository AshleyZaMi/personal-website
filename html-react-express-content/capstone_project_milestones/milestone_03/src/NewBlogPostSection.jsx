import './NewBlogPostSection.css'
import SectionName from './SectionName.jsx'
import PinkField from './PinkField.jsx'
import Button from './Button.jsx'

export default function NewBlogPostSection(props){
    return(
        <>
        <div id='new-blog-post-wrapper'>
            <SectionName
            sectionName='New Blog Post'
            >
            </SectionName>
            {/* the PinkField for inputting the blog name */}
            <PinkField
            pinkStyling='new-blog-pink-field-styling'
            >   
                {/* inside the PinkField are children */}
                <label htmlFor="new-blog-name">Blog name: </label>
                <input type='text' id="new-blog-name"></input>
            </PinkField>

            {/*  the PinkField for inputting the actual blog content*/}
           <PinkField
           pinkStyling='pink-field-content-styling'
           >
            <textarea id='blog-content' className='blog-text-area-styling'></textarea>
           </PinkField>
            <div id='post-button-wrapper'>
                <Button
                className = 'post-button-styling'
                buttonText = 'Post'
                >
                </Button>
            </div>
        </div>
        </>
    )
}
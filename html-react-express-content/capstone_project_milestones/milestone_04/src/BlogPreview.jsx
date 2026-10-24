import './BlogPreview.css'
import Button from './Button.jsx'

export default function BlogPreview(props){
    return(
        <>
        <div id='blog-preview-wrapper'>
            ブログ{props.id}: {props.blogName}
        
        <div id='continue-reading-button-wrapper'>
            <Button
            className='continue-reading-button'
            buttonText = '読み続き'
            ></Button>
        </div>
        </div>
        </>
    )
}
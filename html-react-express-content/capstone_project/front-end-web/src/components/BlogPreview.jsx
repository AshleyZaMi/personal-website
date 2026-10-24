import '../styles/BlogPreview.css'
import Button from './Button.jsx'
import {useNavigate} from 'react-router-dom'

export default function BlogPreview(props){
    // using navigate in order to go to another route in react
    const navigate = useNavigate()
    return(
        <>
        <div id='blog-preview-wrapper'>
            ブログ{props.id}: {props.blogName}
        
        <div id='continue-reading-button-wrapper'>
            <Button
                className='continue-reading-button'
                buttonText = '読み続き'
                onClick={() => navigate(`/blog/${props.id}`)}//here navigating to detail page of blog
            ></Button>
        </div>
        </div>
        </>
    )
}
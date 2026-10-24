import '../styles/NewBlogPostSection.css'
import SectionName from '../components/SectionName.jsx'
import PinkField from '../components/PinkField.jsx'
import Button from '../components/Button.jsx'
import {useState} from 'react'

// will have a live one now so will replace to this path
const API_BASE_URL = import.meta.env.VITE_API_URL

export default function NewBlogPostSection(props){
    const [title, setTitle] = useState('')
    const [content, setContent] = useState('')
    const [error, setError] = useState('')

    const handlePost = async () => {
        setError('')

        if (!title.trim() || !content.trim()) {
            setError('Title and content are required')
            return
        }

        try {
            const res = await fetch(API_BASE_URL + '/api/posts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ title, content })
            })

            const newPost = await res.json()

            if (!res.ok) {
                setError(newPost.message)
                return
            }

            props.onNewPost(newPost) // tell parent about new post
            setTitle('')       // clear form
            setContent('')
        } catch(err) {
            console.log(err)
            setError('Cannot connect to server')
        }
    }
    return(
        <>
        <div id='new-blog-post-wrapper'>
            <SectionName
                sectionName='New Blog Post'
            />

            {/* showing error message here */}
            {error && <div className='text-danger'>{error}</div>}
            
            {/* the PinkField for inputting the blog name */}
            <PinkField
                pinkStyling='new-blog-pink-field-styling'
            >   
                {/* inside the PinkField are children */}
                <label htmlFor="new-blog-name">Blog name: </label>
                <input 
                    type='text' 
                    id="new-blog-name"
                    value={title}
                    onChange={(event) => setTitle(event.target.value)}
                />
            </PinkField>

            {/*  the PinkField for inputting the actual blog content*/}
           <PinkField
           pinkStyling='pink-field-content-styling'
           >
            <textarea 
                id='blog-content' 
                className='blog-text-area-styling'
                value={content}
                onChange={(event) => setContent(event.target.value)}
            />
           </PinkField>

            <div id='post-button-wrapper'>
                <Button
                    className = 'post-button-styling'
                    buttonText = 'Post'
                    onClick={handlePost}
                >
                </Button>
            </div>
        </div>
        </>
    )
}
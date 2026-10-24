import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import '../styles/BlogPostDetail.css'

// will have a live one now so will replace to this path
const API_BASE_URL = import.meta.env.VITE_API_URL

export default function BlogPostDetail() {
    const { id } = useParams() // gets the id from the URL
    const [post, setPost] = useState(null)
    const [error, setError] = useState('')

    useEffect(() => {
        fetch(`${API_BASE_URL}/api/posts/${id}`, {
            credentials: 'include'
        })
        .then(res => res.json())
        .then(data => setPost(data))
        .catch(err => console.log(err))
    }, [id])

    if (!post) return <div>Loading...</div>

    return(
        <div id='blog-detail-wrapper'>
            <div id='blog-display-title'>
                <h1>Title: {post.title}</h1>
                 <p>Date Published: {post.created_at}</p>
            </div>
            <div id='blog-display-content'>
                <p>{post.content}</p>
            </div>
        </div>
    )
}
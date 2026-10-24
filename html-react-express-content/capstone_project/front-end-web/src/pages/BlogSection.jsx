import { useState, useEffect } from 'react'
import NewBlogPostSection from './NewBlogPostSection.jsx'
import PastPostsSection from './PastPostsSection.jsx'

// will have a live one now so will replace to this path
const API_BASE_URL = import.meta.env.VITE_API_URL

export default function BlogSection() {
    const [posts, setPosts] = useState([])

    // fetch posts on load
    useEffect(() => {
        fetch(API_BASE_URL + '/api/posts', { credentials: 'include' })
            .then(res => res.json())
            .then(data => {
                if (Array.isArray(data)) setPosts(data)
                else setPosts([])
            })
            .catch(err => console.log(err))
    }, [])

    const handleNewPost = (newPost) => {
        setPosts(prev => [newPost, ...prev]) // add to top
    }

    return(
        <>
            <NewBlogPostSection onNewPost={handleNewPost} />
            <PastPostsSection posts={posts} />
        </>
    )
}
import '../styles/PastPostsSection.css'
import SectionName from '../components/SectionName.jsx'
import BlogPreview from '../components/BlogPreview.jsx'

export default function PastPostsSection(props){
    return(
        <>
        <div id='past-posts-wrapper'>
            <SectionName
                className = 'past-posts-section-name-styling'
                sectionName='Past Posts'
            />
            
            <div id='posts-container'>
                {props.posts.map(post => (
                    <BlogPreview
                        key={post.post_id}
                        id={post.post_id}
                        blogName = {post.title}
                    />
                ))}
                {/* <BlogPreview
                id={1}
                blogName = '私の国では日本と同じようにふフアッションに季節を大切にする'
                >
                </BlogPreview>
                <BlogPreview
                id={2}
                blogName = '日本人の若者で現在的な人気がある本について'
                >
                </BlogPreview>
                <BlogPreview
                id={3}
                blogName = '日本に気になるニュース'
                >
                </BlogPreview> */}

            </div>

        </div>
        </>
    )
}
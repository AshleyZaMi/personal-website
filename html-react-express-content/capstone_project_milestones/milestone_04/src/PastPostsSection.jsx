import './PastPostsSection.css'
import SectionName from './SectionName.jsx'
import BlogPreview from './BlogPreview.jsx'

export default function PastPostsSection(props){
    return(
        <>
        <div id='past-posts-wrapper'>
            <SectionName
            className = 'past-posts-section-name-styling'
            sectionName='Past Posts'
            >
            </SectionName>
            <div id='posts-container'>
                <BlogPreview
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
                </BlogPreview>

            </div>

        </div>
        </>
    )
}
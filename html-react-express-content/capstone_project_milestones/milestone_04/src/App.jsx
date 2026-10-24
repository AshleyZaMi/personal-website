// external libraries on top
import {HelmetProvider} from "react-helmet-async" //this is for dynamic meta tags depending on page
import {HashRouter, Routes, Route} from 'react-router-dom'

// internal libraries here
import './App.css'
import Navbar from './Navbar.jsx'
import BackgroundOverlay from './BackgroundOverlay.jsx'
import TaskManager from './TaskManager.jsx'
import QuoteSection from './QuoteSection.jsx'
import PomodormoSection from './PomodormoSection.jsx'
import SiteName from './SiteName.jsx'
import Footer from './Footer.jsx'
import TopicRecSection from './TopicRecSection.jsx'
import NewBlogPostSection from './NewBlogPostSection.jsx'
import PastPostsSection from './PastPostsSection.jsx'
import SearchEngineOptimization from './SearchEngineOptimization.jsx'
import SentenceRecSection from './SentenceRecSection.jsx'
import PosSection from './PosSection.jsx'

function App() {
  
  return (
    // /*This is to have dynamic meta tags for each page*/
    <HelmetProvider> 
      {/* // Have to put the whole application inside of the HashRouter
      //HashRouter can display things or remove things, depending on quote
      // on quote "what page" we want it to use */}
      <HashRouter>
        <>
      {/* we need both of these to be in our application by defualt
      even when we're switching "pages" */}
      <BackgroundOverlay/>
      <Navbar></Navbar>
      {/* want to define our pages. Each page is going to be inside of a Route component */}
      <Routes>
        {/* first page (Home page) */}
        <Route path='/' element={
          <>
          <SearchEngineOptimization 
          title="Home" 
          description="Website where Japanese Language learners can manage their studying tasks. Made in order to improve motivation and a platform that can be a really useful tool when studying Japanese. A language learner's dream workspace."/>
          <TaskManager></TaskManager>
          <QuoteSection></QuoteSection>
          <PomodormoSection></PomodormoSection>
           <SiteName
           siteNameStyling='first-page-site-name-styling'
           ></SiteName>
          </>
          
        }/>
        {/* Second Page (blog page) */}
        <Route path='/blog' element={
          <>
          <SearchEngineOptimization 
          title="blog" 
          description="Here language learners can easily get a random blog topic generated, and have a place where they can write their blog. They can also see the past blogs they have written. Its a place where they can utilize in order to enhance their studying by producing writing in that random topic."/>
          <TopicRecSection></TopicRecSection>
          <NewBlogPostSection></NewBlogPostSection>
          <PastPostsSection></PastPostsSection>
           <SiteName></SiteName>
          </>
        }/>

        {/* Third page (Analysis) */}
        <Route path='/analysis' element={
          <>
          <SearchEngineOptimization 
          title="analysis" 
          description="Here language learners can easily break down Part of Speech Components from their sentence in order to understand sentence compoenents. When they click POS it then below it will be the Parts of Speech of that sentence using the spacy NLP library."/>
          <SentenceRecSection></SentenceRecSection>
          <PosSection></PosSection>
           <SiteName
           siteNameStyling='third-page-site-name-styling'
           ></SiteName>
          </>
        }/>
        
      </Routes>
      {/* want these to be visible to all pages */}
      {/* check the site name position might be reliant on others right now */}
     
      <Footer></Footer>
        </>
      </HashRouter> {/*//HashRouter ends here*/}
    </HelmetProvider>
  )
}

export default App

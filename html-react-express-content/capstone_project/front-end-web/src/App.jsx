// external libraries on top
import { HelmetProvider } from "react-helmet-async" //this is for dynamic meta tags depending on page
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom'


// internal libraries here
import './styles/App.css'
import Navbar from './components/Navbar.jsx'
import BackgroundOverlay from './components/BackgroundOverlay.jsx'
import TaskManager from './pages/TaskManager.jsx'
import QuoteSection from './pages/QuoteSection.jsx'
import PomodormoSection from './pages/PomodormoSection.jsx'
import SiteName from './components/SiteName.jsx'
import Footer from './components/Footer.jsx'
import TopicRecSection from './pages/TopicRecSection.jsx'
import SearchEngineOptimization from './components/SearchEngineOptimization.jsx'
import SentenceRecSection from './pages/SentenceRecSection.jsx'
import PosSection from './pages/PosSection.jsx'
import LoginForm from './pages/LoginForm.jsx'
import WelcomeText from './components/WelcomeText.jsx'
import Header from './components/Header.jsx'
import RegisterForm from './pages/RegisterForm.jsx'
import BlogSection from './pages/BlogSection.jsx'
import BlogPostDetail from './pages/BlogPostDetail.jsx'


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
          <BackgroundOverlay />

          {/* want to define our pages. Each page is going to be inside of a Route component */}
          <Routes>
            {/* redirect root to login */}
            <Route path='/' element={<Navigate to='/login' />} />

            {/* first page (Home page) */}
            <Route path='/home' element={
              <>
                <Navbar></Navbar>
                <SearchEngineOptimization
                  title="Home"
                  description="Website where Japanese Language learners can manage their studying tasks. Made in order to improve motivation and a platform that can be a really useful tool when studying Japanese. A language learner's dream workspace." />
                <TaskManager></TaskManager>
                <QuoteSection></QuoteSection>
                <PomodormoSection></PomodormoSection>
                <SiteName
                  siteNameStyling='first-page-site-name-styling'
                ></SiteName>
              </>

            } />
            {/* Second Page (blog page) */}
            <Route path='/blog' element={
              <>
                <Navbar></Navbar>
                <SearchEngineOptimization
                  title="blog"
                  description="Here language learners can easily get a random blog topic generated, and have a place where they can write their blog. They can also see the past blogs they have written. Its a place where they can utilize in order to enhance their studying by producing writing in that random topic." />
                <TopicRecSection></TopicRecSection>
                <BlogSection></BlogSection>
                <SiteName></SiteName>
              </>
            } />

            {/* adding a Blog Post Detail page to see the individual blog */}
            <Route path='/blog/:id' element={
              <>
                <Navbar />
                <BlogPostDetail></BlogPostDetail>
                <SiteName
                siteNameStyling='second-page-site-name-styling'
                ></SiteName>
              </>
            } />

            {/* Third page (Analysis) */}
            <Route path='/analysis' element={
              <>
                <Navbar></Navbar>
                <SearchEngineOptimization
                  title="analysis"
                  description="Here language learners can easily break down Part of Speech Components from their sentence in order to understand sentence compoenents. When they click POS it then below it will be the Parts of Speech of that sentence using the spacy NLP library." />
                <SentenceRecSection></SentenceRecSection>
                <PosSection></PosSection>
                <SiteName
                  siteNameStyling='third-page-site-name-styling'
                ></SiteName>
              </>
            } />
            {/* path for login */}
            <Route path='/login' element={
              <>
                <SearchEngineOptimization
                  title="login"
                  description="Here language learners can easily break down Part of Speech Components from their sentence in order to understand sentence compoenents. When they click POS it then below it will be the Parts of Speech of that sentence using the spacy NLP library." />
                <Header></Header>
                <WelcomeText
                  write="さあ、頑張ろう！"
                ></WelcomeText>
                <LoginForm></LoginForm>
                <SiteName
                  siteNameStyling='login-page-site-name-styling'
                ></SiteName>
              </>
            } />
            {/* path for register */}
            <Route path='/register' element={
              <>
                <SearchEngineOptimization
                  title="register"
                  description="Here language learners can easily break down Part of Speech Components from their sentence in order to understand sentence compoenents. When they click POS it then below it will be the Parts of Speech of that sentence using the spacy NLP library." />
                <Header></Header>
                <WelcomeText
                  write="じゃあ、作ろう！"
                ></WelcomeText>
                <RegisterForm></RegisterForm>
                <SiteName
                  siteNameStyling='register-page-site-name-styling'
                ></SiteName>
              </>
            } />

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

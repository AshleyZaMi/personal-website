import './App.css'
import Navbar from './Navbar.jsx'
import BackgroundOverlay from './BackgroundOverlay.jsx'
import TaskManager from './TaskManager.jsx'
import QuoteSection from './QuoteSection.jsx'
import PomodormoSection from './PomodormoSection.jsx'
import SiteName from './SiteName.jsx'
import Footer from './Footer.jsx'

function App() {
  
  return (
    <>
    <BackgroundOverlay/>
    <Navbar></Navbar>
    <TaskManager></TaskManager>
    <QuoteSection></QuoteSection>
    <PomodormoSection></PomodormoSection>
    <SiteName></SiteName>
    <Footer></Footer>
    </>
  )
}

export default App

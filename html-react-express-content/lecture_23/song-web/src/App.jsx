
import { HashRouter, Routes, Route } from 'react-router-dom'
import Home from './views/Home.jsx'
import SearchForm from './views/SearchForm.jsx';
import AddForm from './views/AddForm.jsx';
import EditForm from './views/EditForm.jsx';
import SearchResults from './views/SearchResults.jsx';
import Track from './views/Track.jsx';
import AddConfirm from './views/AddConfirm.jsx';
import EditConfirm from './views/EditConfirm.jsx'
import Delete from './views/Delete.jsx'

import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.min.js';
import './App.css'

function App() {

  return (
    <HashRouter>

      <Routes>
        <Route path="/" element={<Home />}></Route>
        <Route path="/searchForm" element={<SearchForm />}></Route>
        <Route path="/addForm" element={<AddForm />}></Route>
        <Route path="/editForm/:id" element={<EditForm />}></Route>
        <Route path="/searchResults" element={<SearchResults />}></Route>
        <Route path="/track/:id" element={<Track />}></Route>
        <Route path="/addConfirm" element={<AddConfirm />}></Route>
        <Route path="/editConfirm" element={<EditConfirm />}></Route>
        <Route path="/delete" element={<Delete />}></Route>
      </Routes>

    </ HashRouter>
  )
}

export default App

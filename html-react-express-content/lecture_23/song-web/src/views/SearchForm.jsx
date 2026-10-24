import { useState, useEffect } from "react" //useEffect will allow us to just do one call once at the very beginning
import { Link, useNavigate } from 'react-router-dom'
import {fetchGenres, fetchMediaTypes} from '../fetch/index.js' //want object destructing because return 

export default function SearchForm() {

	const [genres, setGenres] = useState([
		{ genreId: 1, name: 'Rock' },
		{ genreId: 2, name: 'Jazz' }
	])
	const [mediaTypes, setMediaTypes] = useState([
		{ mediaTypeId: 1, name: 'MPEG audio file' },
		{ mediaTypeId: 2, name: 'Protected AAC audio file' }
	])
	const [trackInput, setTrackInput] = useState('')
	const [genreInput, setGenreInput] = useState('')
	const [mediaInput, setMediaInput] = useState('')
	const navigate = useNavigate()

	// does create an overhead but its cleaner
	// this function is calling fetchGenres from fetch
	const getGenres = async()=>{
		// calling the fetch files, and can set the set genre to be that fetch file
		setGenres(await fetchGenres())
	}

	const getMediaTypes = async()=>{
		setMediaTypes(await fetchMediaTypes())
	}

	// useEffect(function, array)
	useEffect(() => {
		getGenres()
		getMediaTypes()
	}, [])

	

	// useEffect(function, array)
	

	const handleSubmit = (event) => {
		event.preventDefault()

		navigate("/searchResults", { state: { trackInput, genreInput, mediaInput } })
	}

	return (
		<>
			<ol className="breadcrumb p-3">
				<li className="breadcrumb-item"><Link to="/">Home</Link></li>
				<li className="breadcrumb-item active">Search</li>
			</ol>
			<div className="container">
				<div className="row">
					<h1 className="col-12 mt-4 mb-4">Song Search Form</h1>
				</div>
			</div>
			<div className="container">

				<form onSubmit={handleSubmit}>

					<div className="form-group row mb-3">
						<label htmlFor="name-id" className="col-sm-3 col-form-label text-sm-right">Track Name:</label>
						<div className="col-sm-9">
							<input value={trackInput} onChange={(e) => setTrackInput(e.target.value)} type="text" className="form-control" id="name-id" name="trackName" />
						</div>
					</div>
					<div className="form-group row mb-3">
						<label htmlFor="genre-id" className="col-sm-3 col-form-label text-sm-right">Genre:</label>
						<div className="col-sm-9">
							<select value={genreInput} onChange={(e) => setGenreInput(e.target.value)} name="genreId" id="genre-id" className="form-control">
								<option value="">-- All --</option>

								{genres.map((row) => {
									return (
										<option key={row.genreId} value={row.genreId}>
											{row.name}
										</option>
									)
								})}

							</select>
						</div>
					</div>
					<div className="form-group row mb-3">
						<label htmlFor="media-type-id" className="col-sm-3 col-form-label text-sm-right">Media Type:</label>
						<div className="col-sm-9">
							<select value={mediaInput} onChange={(e) => setMediaInput(e.target.value)} name="mediaTypeId" id="media-type-id" className="form-control">
								<option value="" >-- All --</option>

								{mediaTypes.map((row) => {
									return (
										<option key={row.mediaTypeId} value={row.mediaTypeId}>
											{row.name}
										</option>
									)
								})}

							</select>
						</div>
					</div>
					<div className="form-group row">
						<div className="col-sm-3"></div>
						<div className="col-sm-9 mt-2">
							<button type="submit" className="btn btn-primary me-3">Search</button>
							<button type="reset" className="btn btn-light">Reset</button>
						</div>
					</div>
				</form>
			</div>
		</>
	)
}
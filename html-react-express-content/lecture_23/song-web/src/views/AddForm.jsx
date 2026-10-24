import { useState, useEffect } from "react"
import { useNavigate, Link } from "react-router-dom"
import {fetchGenres, fetchAlbums, fetchMediaTypes} from '../fetch/index.js'

export default function AddForm() {
	const [genres, setGenres] = useState([
		{ genreId: 1, name: 'Rock' },
		{ genreId: 2, name: 'Jazz' }
	])
	const [mediaTypes, setMediaTypes] = useState([
		{ mediaTypeId: 1, name: 'MPEG audio file' },
		{ mediaTypeId: 2, name: 'Protected AAC audio file' }
	])
	const [albums, setAlbums] = useState([
		{ albumId: 1, name: "For Those About To Rock (We Salute You)" }
	])
	// state for user input
	const [trackInput, setTrackInput] = useState('')
	const [mediaInput, setMediaInput] = useState('')
	const [genreInput, setGenreInput] = useState('')
	const [millisecondsInput, setMillisecondsInput] = useState('')
	const [priceInput, setPriceInput] = useState('')
	const [albumInput, setAlbumInput] = useState('')
	const [composerInput, setComposerInput] = useState('')
	const [bytesInput, setBytesInput] = useState('')
	const navigate = useNavigate()

	// create functiinn to get drop downs. setting the genres
	const getDropdowns = async ()=> {
		setGenres(await fetchGenres())
		setAlbums(await fetchAlbums())
		setMediaTypes(await fetchMediaTypes())
	}
	useEffect(()=>{
		getDropdowns()
	}, [])
	const handleSubmit = async (event) => {
		event.preventDefault()

		const fields = {
			trackName: trackInput,
			mediaTypeId: mediaInput,
			genreId: genreInput,
			milliseconds: millisecondsInput,
			price: priceInput,
			albumId: albumInput,
			bytes: bytesInput,
			composer: composerInput
		}

		const response = await fetch("http://localhost:3000/tracks", {
			method: "POST",
			body: new URLSearchParams(fields)
		})
		const result = await response.json()

		navigate('/addConfirm', {
			state: {
				status: response.status,
				message: result.message,
				trackName: trackInput
			}
		})
	}

	return (
		<>

			<ol className="breadcrumb p-3">
				<li className="breadcrumb-item"><Link to="/">Home</Link></li>
				<li className="breadcrumb-item active">Add</li>
			</ol>
			<div className="container">
				<div className="row">
					<h1 className="col-12 mt-4 mb-4">Add a Song</h1>
				</div>
			</div>
			<div className="container">

				<form onSubmit={handleSubmit}>

					<div className="form-group row mb-3">
						<label htmlFor="name-id" className="col-sm-3 col-form-label text-sm-right">
							Track Name: <span className="text-danger">*</span>
						</label>
						<div className="col-sm-9">
							<input value={trackInput} onChange={(e) => setTrackInput(e.target.value)} type="text" className="form-control" id="name-id" name="trackName" />
						</div>
					</div>

					<div className="form-group row mb-3">
						<label htmlFor="media-type-id" className="col-sm-3 col-form-label text-sm-right">
							Media Type: <span className="text-danger">*</span>
						</label>
						<div className="col-sm-9">
							<select value={mediaInput} onChange={(e) => setMediaInput(e.target.value)} name="mediaTypeId" id="media-type-id" className="form-control">
								<option value="" disabled>-- Select One --</option>

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

					<div className="form-group row mb-3">
						<label htmlFor="genre-id" className="col-sm-3 col-form-label text-sm-right">
							Genre: <span className="text-danger">*</span>
						</label>
						<div className="col-sm-9">
							<select value={genreInput} onChange={(e) => setGenreInput(e.target.value)} name="genreId" id="genre-id" className="form-control">
								<option value="" disabled>-- Select One --</option>

								{genres.map((row) => {
									return (
										<option key={row.genreId} value={row.genreId}>
											{row.name}
										</option>
									)
								})}Í

							</select>
						</div>
					</div>

					<div className="form-group row mb-3">
						<label htmlFor="milliseconds-id" className="col-sm-3 col-form-label text-sm-right">
							Milliseconds: <span className="text-danger">*</span>
						</label>
						<div className="col-sm-9">
							<input value={millisecondsInput} onChange={(e) => setMillisecondsInput(e.target.value)} type="number" name="milliseconds" min="0" id="milliseconds-id" className="form-control" />
						</div>
					</div>

					<div className="form-group row mb-3">
						<label htmlFor="price-id" className="col-sm-3 col-form-label text-sm-right">
							Price: <span className="text-danger">*</span>
						</label>
						<div className="col-sm-9">
							<input value={priceInput} onChange={(e) => setPriceInput(e.target.value)} type="number" step="0.01" min="0" name="price" id="price-id" className="form-control" />
						</div>
					</div>

					<div className="form-group row mb-3">
						<label htmlFor="album-id" className="col-sm-3 col-form-label text-sm-right">
							Album:
						</label>
						<div className="col-sm-9">
							<select value={albumInput} onChange={(e) => setAlbumInput(e.target.value)} name="albumId" id="album-id" className="form-control">
								<option value="" disabled>-- Select One --</option>

								{albums.map((row) => {
									return (
										// here be careful what the RESTAPI displays
										<option key={row.albumId} value={row.albumId}>
											{row.title}
										</option>
									)

								})}

							</select>
						</div>
					</div>

					<div className="form-group row mb-3">
						<label htmlFor="composer-id" className="col-sm-3 col-form-label text-sm-right">
							Composer:
						</label>
						<div className="col-sm-9">
							{/* Note: Need on change, and also need a corresponding value, in order to change that value */}
							<input value={composerInput} onChange={(e) => setComposerInput(e.target.value)} type="text" name="composer" id="composer-id" className="form-control" />
						</div>
					</div>

					<div className="form-group row mb-3">
						<label htmlFor="bytes-id" className="col-sm-3 col-form-label text-sm-right">
							Bytes:
						</label>
						<div className="col-sm-9">
							<input value={bytesInput} onChange={(e) => setBytesInput(e.target.value)} type="number" name="bytes" min="0" id="bytes-id" className="form-control" />
						</div>
					</div>

					<div className="form-group row mb-3">
						<div className="ml-auto col-sm-9">
							<span className="text-danger font-italic">* Required</span>
						</div>
					</div>

					<div className="form-group row">
						<div className="col-sm-3"></div>
						<div className="col-sm-9 mt-2">
							<button type="submit" className="btn btn-primary me-3">Submit</button>
							<button type="reset" className="btn btn-light">Reset</button>
						</div>
					</div>
				</form>
			</div>
		</>
	)
}


import { useState } from "react"
import { Link, useParams } from "react-router-dom"
import Table from 'react-bootstrap/Table'
import { useEffect } from "react"

export default function Track() {

	const [track, setTrack] = useState({
		name: "For Those About To Rock (We Salute You)",
		composer: "Angus Young, Malcolm Young, Brian Johnson",
		genre: "Rock",
		artist: "AC/DC",
		mediaType: "MPEG audio file",
		album: "For Those About To Rock (We Salute You)",
		milliseconds: 343719,
		bytes: 11170334,
		price: 0.99
	})
	const params = useParams()

	const fetchTrack = async () => {
		const response = await fetch("http://localhost:3000/tracks/" + params.id)
		const result = await response.json()
		setTrack(result)
	}

	useEffect(() => {
		fetchTrack()
	}, [params.id])

	return (
		<>
			<ol className="breadcrumb p-3">
				<li className="breadcrumb-item"><Link to="/">Home</Link></li>
				<li className="breadcrumb-item"><Link to="/searchForm">Search</Link></li>
				<li className="breadcrumb-item"><Link to="/searchResults">Results</Link></li>
				<li className="breadcrumb-item active">Details</li>
			</ol>
			<div className="container">
				<div className="row">
					<h1 className="col-12 mt-4">Song Details</h1>
				</div>
			</div>
			<div className="container">
				<div className="row mt-4">
					<div className="col-12">

						<Table responsive>
							<tbody>
								<tr>
									<th className="text-right">Track Name:</th>
									<td>{track.name}</td>
								</tr>
								<tr>
									<th className="text-right">Artist Name:</th>
									<td>{track.artist}</td>
								</tr>
								<tr>
									<th className="text-right">Composer:</th>
									<td>{track.composer}</td>
								</tr>
								<tr>
									<th className="text-right">Album:</th>
									<td>{track.album}</td>
								</tr>
								<tr>
									<th className="text-right">Genre:</th>
									<td>{track.genre}</td>
								</tr>
								<tr>
									<th className="text-right">Milliseconds:</th>
									<td>{track.milliseconds}</td>
								</tr>
								<tr>
									<th className="text-right">Bytes:</th>
									<td>{track.bytes}</td>
								</tr>
								<tr>
									<th className="text-right">Price:</th>
									<td>{track.price}</td>
								</tr>
								<tr>
									<th className="text-right">Media Type:</th>
									<td>{track.mediaType}</td>
								</tr>
							</tbody>
						</Table>

					</div>
				</div>
				<div className="row mt-4 mb-4">
					<div className="col-12">
						<Link to="/searchResults" role="button" className="btn btn-primary">Back to Search Results</Link>
					</div>
				</div>
			</div>
		</>
	)
}


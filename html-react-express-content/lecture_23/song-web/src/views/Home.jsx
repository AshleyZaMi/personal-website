import { Link } from "react-router-dom"

export default function Home() {
	return (
		<>
			<div className="container">
				<div className="row">
					<h1 className="col-12 mt-4 mb-4">Song Databse</h1>
				</div>
			</div>
			<div className="container">
				<div className="row">
					<div className="col-12 col-md-6 d-grid gap-2">
						<Link to="/searchForm" className="btn btn-primary btn-lg mt-4 mt-md-2" role="button">Search for Songs</Link>
					</div>
					<div className="col-12 col-md-6 d-grid gap-2">
						<Link to="/addForm" className="btn btn-primary btn-lg mt-4 mt-md-2" role="button">Add a Song</Link>
					</div>
				</div>
			</div>
		</>
	)
}

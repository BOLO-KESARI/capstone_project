"""AviationStack API Client – wraps every public endpoint."""

from __future__ import annotations

from typing import Any, Dict, Optional

import os
import httpx

from config import API_ACCESS_KEY, BASE_URL


class AviationStackClient:
    """Lightweight async client for the AviationStack REST API."""

    def __init__(self, access_key: Optional[str] = None):
        self.access_key = access_key or API_ACCESS_KEY
        self.base_url = BASE_URL

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    async def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a GET request and return the JSON response."""
        params = params or {}
        params["access_key"] = self.access_key
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.base_url}/{endpoint}", params=params
                )
                # If API returns 4xx or 5xx, we catch it
                if response.status_code != 200:
                    return {
                        "error": {
                            "code": f"HTTP_{response.status_code}",
                            "message": response.text or "API Error"
                        },
                        "data": []
                    }
                return response.json()
        except Exception as e:
            return {
                "error": {
                    "code": "CLIENT_ERROR",
                    "message": str(e)
                },
                "data": []
            }

    # ------------------------------------------------------------------ #
    #  Public endpoints
    # ------------------------------------------------------------------ #

    # 1. Flights (real-time & historical)
    async def get_flights(
        self,
        limit: int = 100,
        offset: int = 0,
        flight_status: Optional[str] = None,
        flight_date: Optional[str] = None,
        dep_iata: Optional[str] = None,
        arr_iata: Optional[str] = None,
        dep_icao: Optional[str] = None,
        arr_icao: Optional[str] = None,
        airline_name: Optional[str] = None,
        airline_iata: Optional[str] = None,
        flight_number: Optional[str] = None,
        flight_iata: Optional[str] = None,
        min_delay_dep: Optional[int] = None,
        min_delay_arr: Optional[int] = None,
        max_delay_dep: Optional[int] = None,
        max_delay_arr: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Look up real-time / historical flights."""
        return await self._get(
            "flights",
            {
                "limit": limit,
                "offset": offset,
                "flight_status": flight_status,
                "flight_date": flight_date,
                "dep_iata": dep_iata,
                "arr_iata": arr_iata,
                "dep_icao": dep_icao,
                "arr_icao": arr_icao,
                "airline_name": airline_name,
                "airline_iata": airline_iata,
                "flight_number": flight_number,
                "flight_iata": flight_iata,
                "min_delay_dep": min_delay_dep,
                "min_delay_arr": min_delay_arr,
                "max_delay_dep": max_delay_dep,
                "max_delay_arr": max_delay_arr,
            },
        )

    # 2. Routes
    async def get_routes(
        self,
        limit: int = 100,
        offset: int = 0,
        dep_iata: Optional[str] = None,
        arr_iata: Optional[str] = None,
        dep_icao: Optional[str] = None,
        arr_icao: Optional[str] = None,
        airline_iata: Optional[str] = None,
        airline_icao: Optional[str] = None,
        flight_number: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Look up airline routes."""
        return await self._get(
            "routes",
            {
                "limit": limit,
                "offset": offset,
                "dep_iata": dep_iata,
                "arr_iata": arr_iata,
                "dep_icao": dep_icao,
                "arr_icao": arr_icao,
                "airline_iata": airline_iata,
                "airline_icao": airline_icao,
                "flight_number": flight_number,
            },
        )

    # 3. Airports
    async def get_airports(
        self,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Look up global airports."""
        return await self._get(
            "airports",
            {"limit": limit, "offset": offset, "search": search},
        )

    # 4. Airlines
    async def get_airlines(
        self,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Look up global airlines."""
        return await self._get(
            "airlines",
            {"limit": limit, "offset": offset, "search": search},
        )

    # 5. Airplanes
    async def get_airplanes(
        self,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Look up airplanes / aircraft."""
        return await self._get(
            "airplanes",
            {"limit": limit, "offset": offset, "search": search},
        )

    # 6. Aircraft Types
    async def get_aircraft_types(
        self,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Look up aircraft types."""
        return await self._get(
            "aircraft_types",
            {"limit": limit, "offset": offset, "search": search},
        )

    # 7. Aviation Taxes
    async def get_taxes(
        self,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Look up aviation taxes."""
        return await self._get(
            "taxes",
            {"limit": limit, "offset": offset, "search": search},
        )

    # 8. Cities
    async def get_cities(
        self,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Look up global cities."""
        return await self._get(
            "cities",
            {"limit": limit, "offset": offset, "search": search},
        )

    # 9. Countries
    async def get_countries(
        self,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Look up countries."""
        return await self._get(
            "countries",
            {"limit": limit, "offset": offset, "search": search},
        )

    # 10. Flight Schedules (Timetable)
    async def get_timetable(
        self,
        iata_code: Optional[str] = None,
        icao_code: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Look up scheduled flights (timetable)."""
        return await self._get(
            "timetable",
            {
                "iata_code": iata_code,
                "icao_code": icao_code,
                "limit": limit,
                "offset": offset,
            },
        )

    # 11. Flight Future Schedules
    async def get_flight_future(
        self,
        iata_code: Optional[str] = None,
        icao_code: Optional[str] = None,
        date: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Look up future flight schedules."""
        return await self._get(
            "flightsFuture",
            {
                "iata_code": iata_code,
                "icao_code": icao_code,
                "date": date,
                "limit": limit,
                "offset": offset,
            },
        )

    # 12. Live Weather (OpenWeather)
    async def get_weather(self, iata_code: str) -> Dict[str, Any]:
        """Fetch live weather for an airport."""
        indian_airports = {
            "DEL": {"lat": 28.5562, "lon": 77.1000},
            "BOM": {"lat": 19.0896, "lon": 72.8656},
            "BLR": {"lat": 13.1986, "lon": 77.7066},
            "MAA": {"lat": 12.9941, "lon": 80.1709},
            "HYD": {"lat": 17.2403, "lon": 78.4298},
            "CCU": {"lat": 22.6547, "lon": 88.4467},
            "AMD": {"lat": 23.0734, "lon": 72.6347},
            "PNQ": {"lat": 18.5822, "lon": 73.9197},
            "GOI": {"lat": 15.3800, "lon": 73.8300}
        }
        
        coords = indian_airports.get(iata_code)
        if not coords:
            return {"error": {"message": "Weather only available for main Indian hubs."}}
            
        weather_key = os.getenv("OPENWEATHER_API_KEY", "")
        url = "https://api.openweathermap.org/data/2.5/weather"
        
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, params={
                    "lat": coords["lat"], "lon": coords["lon"],
                    "appid": weather_key, "units": "metric"
                })
                if resp.status_code != 200:
                    return {"error": {"message": f"API Error ({resp.status_code})"}}
                
                data = resp.json()
                wind_kmph = data["wind"]["speed"] * 3.6
                vis_km = data.get("visibility", 10000) / 1000
                
                # Severity Logic
                severity = 1.0
                if wind_kmph > 40: severity += 3.0
                if vis_km < 2: severity += 4.0
                if any(w in data["weather"][0]["main"].lower() for w in ['storm', 'thunderstorm']): severity += 3.0

                return {
                    "data": [{
                        "airport_code": iata_code,
                        "temperature": f"{data['main']['temp']}°C",
                        "feels_like": f"{data['main']['feels_like']}°C",
                        "humidity": f"{data['main']['humidity']}%",
                        "wind_speed": f"{round(wind_kmph, 2)} km/h",
                        "weather_condition": data["weather"][0]["main"],
                        "visibility_km": f"{vis_km} km",
                        "severity_index": round(min(10, severity), 1)
                    }]
                }
            except Exception as e:
                return {"error": {"message": str(e)}}

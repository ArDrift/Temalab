using AdasVetelServer.dbHandler;
using AdasVetelServer.model;
using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace AdasVetelServer.db
{
    class RouteContainer
    {
        public static RouteContainer Instance { get; } = new RouteContainer();

        private readonly List<RouteMapperBase> routeMaps;
        private readonly DbHandler dbh;
        private RouteContainer()
        {
            dbh = new DbHandler();

            routeMaps = new List<RouteMapperBase>()
            {
                new NoIdRouteMapper<Cim>("^/cimek$", dbh.getCimek),
                new NoIdRouteMapper<Gepjarmu>("^/gepjarmuvek$", dbh.getGepjarmuvek),
                new NoIdRouteMapper<Ingatlan>("^/ingatlanok$", dbh.getIngatlanok),
                new NoIdRouteMapper<Ingo>("^/ingok$", dbh.getIngosagok),
                new NoIdRouteMapper<Reszvetel>("^/reszvetelek$", dbh.getReszvetelek),
                new NoIdRouteMapper<Szemely>("^/szemelyek$", dbh.getSzemelyek),
                new NoIdRouteMapper<Szervezet>("^/szervezetek$", dbh.getSzervezetek),
                new NoIdRouteMapper<Szerzodes>("^/szerzodesek$", dbh.getSzerzodesek),
                new NoIdRouteMapper<SzerzodesTargy>("^/szerzodestargyak$", dbh.getSzerzodesTargyak),
                new NoIdRouteMapper<Teljesites>("^/teljesitesek$", dbh.getTeljesitesek),

                new OneIdRouteMapper<Szerzodes>("^/szemelyek/[0-9]+/szerzodesek$", dbh.getSzerzodesekBySzemely),
                new OneIdRouteMapper<Reszvetel>("^/szemelyek/[0-9]+/reszvetelek$", dbh.getReszvetelekBySzemely),
                new OneIdRouteMapper<Cim>("^/szemelyek/[0-9]+/cimek$", dbh.getCimekBySzemely),

                new OneIdRouteMapper<Szerzodes>("^/szervezetek/[0-9]+/szerzodesek$", dbh.getSzerzodesekBySzervezet),
                new OneIdRouteMapper<Reszvetel>("^/szervezetek/[0-9]+/reszvetelek$", dbh.getReszvetelekBySzervezet),
                new OneIdRouteMapper<Cim>("^/szervezetek/[0-9]+/cimek$", dbh.getCimekBySzervezet),

                new OneIdRouteMapper<Szemely>("^/szerzodesek/[0-9]+/szemelyek$", dbh.getSzemelyekBySzerzodes),
                new OneIdRouteMapper<Szervezet>("^/szerzodesek/[0-9]+/szervezetek$", dbh.getSzervezetekBySzerzodes),
                new OneIdRouteMapper<Reszvetel>("^/szerzodesek/[0-9]+/reszvetelek$", dbh.getReszvetelekBySzerzodes),
                new OneIdRouteMapper<Ingatlan>("^/szerzodesek/[0-9]+/ingatlanok$", dbh.getIngatlanokBySzerzodes),
                new OneIdRouteMapper<Gepjarmu>("^/szerzodesek/[0-9]+/gepjarmuvek$", dbh.getGepjarmuvekBySzerzodes),
                new OneIdRouteMapper<Ingo>("^/szerzodesek/[0-9]+/ingok$", dbh.getIngosagokBySzerzodes),
                new OneIdRouteMapper<Teljesites>("^/szerzodesek/[0-9]+/teljesitesek$", dbh.getTeljesitesekBySzerzodes),
                new OneIdRouteMapper<SzerzodesTargy>("^/szerzodesek/[0-9]+/szerzodestargyak$", dbh.getSzerzodesTargyakBySzerzodes),

            };
            dbh.closeConnection();

        }
        public List<byte[]> getDataByRoute(string route)
        {
            dbh.openConnection();

            List<byte[]> res = null;
            foreach (var item in routeMaps)
            {
                if (Regex.IsMatch(route, item.Route))
                    res = item.RunQuery(route);
            }

            dbh.closeConnection();

            if (res != null)
                return res;
            throw new Exception("Invalid route");
        }
    }
}

using AdasVetelServer.model;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace AdasVetelServer.db
{

    public delegate List<T> OneIdCallBack<T>(int? id);
    class OneIdRouteMapper<T> : RouteMapperBase where T : DbElement
    {
        private event OneIdCallBack<T> CallBack;
        public OneIdRouteMapper(string route, OneIdCallBack<T> callback) : base(route)
        {
            CallBack = callback;
        }

        public override List<byte[]> RunQuery(string route)
        {
            int id = int.Parse(Regex.Match(route, "[0-9]+").Value);
            List<byte[]> res = new List<byte[]>();
            foreach (T item in CallBack(id))
                res.Add(JsonSerializer.SerializeToUtf8Bytes(item, new JsonSerializerOptions { WriteIndented = true }));
            return res;
        }
    }
}
